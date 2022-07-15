# coding=utf-8
import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.functional import embedding
from transformers.models.auto.configuration_auto import AutoConfig
from libs.model.extractor import RoiFeatExtraxtor
from libs.configs.default import counter

import detectron2
from .swin_transformer import VisionBackbone

import torch.nn.functional as F
from transformers import AutoModel
from transformers.utils import logging
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
)
from ..layoutlmv2.modeling_layoutlmv2 import LayoutLMv2Layer
from ..layoutlmv2.modeling_layoutlmv2 import *
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMPooler as GraphDocPooler
from .configuration_graphdoc import GraphDocConfig
from .utils import align_logits

logger = logging.get_logger(__name__)

GraphDocLayerNorm = torch.nn.LayerNorm


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.max_positions = int(1e5)

    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        positions
    ):
        self.weights = self.weights.to(positions.device)

        return (
            self.weights[positions.reshape(-1)]
            .view(positions.size() + (-1,))
            .detach()
        )


class Sentence_Embedding(nn.Module):
    """This module produces sentence embeddings of input_ids.

    """

    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.hidden_size
        self.embedding_model = AutoModel.from_pretrained(config.sentence_model).eval()
        sentence_embedding_dim = AutoConfig.from_pretrained(config.sentence_model).hidden_size
        self.transform = nn.Linear(sentence_embedding_dim, self.embedding_dim)

    def forward(self, input_ids, attention_mask, is_target=False, max_inputs=int(1e8)):
        with torch.no_grad():
            B, L, D = input_ids.shape

            # total_sentence_embed = torch.rand((B, L, self.embedding_dim)).to(input_ids.device)

            input_ids = input_ids.reshape(-1, D)
            attention_mask = attention_mask.reshape(-1, D)
            total_sentence_embed = self.embedding_model(input_ids=input_ids.long(), \
                    attention_mask=attention_mask.long()).pooler_output
            total_sentence_embed = total_sentence_embed.reshape(B, L, -1)

            # total_sentence_embed = self.embedding_model(input_ids=input_ids.long(), \
            #         attention_mask=attention_mask.long()).pooler_output

            # start_idx = 0
            # sentence_num = input_ids.shape[0]
            # total_sentence_embed = []
            # while start_idx < sentence_num:
            #     end_idx = min(start_idx + max_inputs, sentence_num)
            #     sentence_embed = self.embedding_model(input_ids=input_ids[start_idx:end_idx].long(), \
            #         attention_mask=attention_mask[start_idx:end_idx].long()).pooler_output
            #     start_idx += max_inputs
            #     total_sentence_embed.append(sentence_embed)

            # total_sentence_embed = torch.cat(total_sentence_embed, dim=0)
            # total_sentence_embed = total_sentence_embed.reshape(B, L, -1)

        if is_target:
            return total_sentence_embed
        else:
            return self.transform(total_sentence_embed)


class GraphDocEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(GraphDocEmbeddings, self).__init__()
        self.sentence_embeddings = Sentence_Embedding(config)
        self.max_2d_position_embeddings = config.max_2d_position_embeddings

        self.use_abs_emb = config.use_abs_emb
        if self.use_abs_emb:
            self.abs_emb_type = config.abs_emb_type
            if self.abs_emb_type == 'Sinusoidal':
                self.expand_wh_scale = config.expand_wh_scale
                self.abs_position_embeddings_transform = nn.Linear(config.hidden_size, config.hidden_size)
                self.position_embeddings = SinusoidalPositionalEmbedding(embedding_dim=config.hidden_size, padding_idx=config.pad_token_id, init_size=config.max_position_embeddings)
                self.abs_position_embeddings = SinusoidalPositionalEmbedding(embedding_dim=config.coordinate_size, padding_idx=config.pad_token_id, init_size=config.max_2d_position_embeddings)
            else:
                self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
                self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
                self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
                self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
                self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)

        self.use_rel_2d = config.use_rel_2d
        if self.use_rel_2d:
            self.rel_topk = config.rel_topk
            self.use_rel_emb = config.use_rel_emb
            if self.use_rel_emb:
                self.rel_position_embeddings = SinusoidalPositionalEmbedding(embedding_dim=config.pos_embed_size, padding_idx=config.pad_token_id)
                self.W_tl = nn.Linear(in_features=int(2*config.pos_embed_size), out_features=config.hidden_size)
                self.W_tr = nn.Linear(in_features=int(2*config.pos_embed_size), out_features=config.hidden_size)
                self.W_bl = nn.Linear(in_features=int(2*config.pos_embed_size), out_features=config.hidden_size)
                self.W_br = nn.Linear(in_features=int(2*config.pos_embed_size), out_features=config.hidden_size)

        self.LayerNorm = GraphDocLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def _cal_spatial_position_embeddings(self, bbox):
        if self.use_abs_emb:
            if self.abs_emb_type == 'Sinusoidal':
                x1 = torch.clamp(bbox[:, :, 0], 0, self.max_2d_position_embeddings-1) # B, L
                y1 = torch.clamp(bbox[:, :, 1], 0, self.max_2d_position_embeddings-1) # B, L
                x2 = torch.clamp(bbox[:, :, 2], 0, self.max_2d_position_embeddings-1) # B, L
                y2 = torch.clamp(bbox[:, :, 3], 0, self.max_2d_position_embeddings-1) # B, L
                w = torch.clamp((x2-x1)*self.expand_wh_scale, 0, self.max_2d_position_embeddings-1).to(bbox.dtype)
                h = torch.clamp((y2-y1)*self.expand_wh_scale, 0, self.max_2d_position_embeddings-1).to(bbox.dtype)

                left_position_embeddings = self.abs_position_embeddings(x1)
                upper_position_embeddings = self.abs_position_embeddings(y1)
                right_position_embeddings = self.abs_position_embeddings(x2)
                lower_position_embeddings = self.abs_position_embeddings(y2)
                w_position_embeddings = self.abs_position_embeddings(w)
                h_position_embeddings = self.abs_position_embeddings(h)

                spatial_position_embeddings = torch.cat(
                    [
                        left_position_embeddings,
                        upper_position_embeddings,
                        right_position_embeddings,
                        lower_position_embeddings,
                        h_position_embeddings,
                        w_position_embeddings,
                    ],
                    dim=-1,
                )
                spatial_position_embeddings = self.abs_position_embeddings_transform(spatial_position_embeddings)
            else:
                try:
                    left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
                    upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
                    right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
                    lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
                except IndexError as e:
                    raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e

                h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
                w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

                spatial_position_embeddings = torch.cat(
                    [
                        left_position_embeddings,
                        upper_position_embeddings,
                        right_position_embeddings,
                        lower_position_embeddings,
                        h_position_embeddings,
                        w_position_embeddings,
                    ],
                    dim=-1,
                )
            return spatial_position_embeddings

        else:
            return None

    def _cal_rel_position_embeddings(self, bbox, bbox_mask):
        if self.use_rel_2d:
            bbox = bbox.masked_fill((1-bbox_mask).unsqueeze(-1).to(torch.bool), int(1e8)) # remove padding token

            _, L, _ = bbox.shape
            topk = min(L-2, self.rel_topk)

            x1 = bbox[:, :, 0] # B, L
            y1 = bbox[:, :, 1] # B, L
            x2 = bbox[:, :, 2] # B, L
            y2 = bbox[:, :, 3] # B, L
            xc = (x1 + x2) // 2
            yc = (y1 + y2) // 2

            # topk index between [CLS] and other bboxes
            cls_bbox = bbox[:, :1]
            cls_xc = (cls_bbox[:, :, 0] + cls_bbox[:, :, 2]) // 2
            cls_yc = (cls_bbox[:, :, 1] + cls_bbox[:, :, 3]) // 2

            diff_xc = cls_xc[:, :, None] - xc[:, None, :] # (B, 1, L)
            diff_yc = cls_yc[:, :, None] - yc[:, None, :] # (B, 1, L)

            distance = diff_xc.pow(2) + diff_yc.pow(2)
            cls_topk_index = distance.topk(topk, dim=-1, largest=False)[1] # (B, 1, topk)

            # topk index between bboxes except [CLS]
            diff_xc = xc[:, 1:, None] - xc[:, None, 1:] # (B, L-1, L-1)
            diff_yc = yc[:, 1:, None] - yc[:, None, 1:] # (B, L-1, L-1)

            distance = diff_xc.pow(2) + diff_yc.pow(2)
            topk_index = distance.topk(topk-1, dim=-1, largest=False)[1] # (B, L-1, topk-1)
            topk_index = topk_index + 1 # cause by shift [CLS]
            topk_index = torch.cat([torch.zeros_like(topk_index[:,:,:1]), topk_index], dim=-1) # append [CLS] token, (B, L-1, topk)

            # concatenate the topk index
            topk_index = torch.cat([cls_topk_index, topk_index], dim=1) # (B, L, topk)

            if self.use_rel_emb:
                # diff
                diff_x1 = x1[:, :, None] - x1[:, None, :] # B, L, L
                diff_y1 = y1[:, :, None] - y1[:, None, :] # B, L, L
                diff_x2 = x2[:, :, None] - x2[:, None, :] # B, L, L
                diff_y2 = y2[:, :, None] - y2[:, None, :] # B, L, L

                diff_x1 = diff_x1.gather(2, topk_index) # B, L, topk
                diff_y1 = diff_y1.gather(2, topk_index) # B, L, topk
                diff_x2 = diff_x2.gather(2, topk_index) # B, L, topk
                diff_y2 = diff_y2.gather(2, topk_index) # B, L, topk

                diff_x1 = torch.clamp(diff_x1, 1-self.max_2d_position_embeddings, self.max_2d_position_embeddings-1) # B, L, topk
                diff_y1 = torch.clamp(diff_y1, 1-self.max_2d_position_embeddings, self.max_2d_position_embeddings-1) # B, L, topk
                diff_x2 = torch.clamp(diff_x2, 1-self.max_2d_position_embeddings, self.max_2d_position_embeddings-1) # B, L, topk
                diff_y2 = torch.clamp(diff_y2, 1-self.max_2d_position_embeddings, self.max_2d_position_embeddings-1) # B, L, topk

                diff_x1 = self.rel_position_embeddings(diff_x1) # B, L, topk, D
                diff_y1 = self.rel_position_embeddings(diff_y1) # B, L, topk, D
                diff_x2 = self.rel_position_embeddings(diff_x2) # B, L, topk, D
                diff_y2 = self.rel_position_embeddings(diff_y2) # B, L, topk, D

                p_tl = self.W_tl(torch.cat([diff_x1, diff_y1], dim=-1)) # B, L, topk, H*D
                p_tr = self.W_tr(torch.cat([diff_x2, diff_y1], dim=-1)) # B, L, topk, H*D
                p_bl = self.W_bl(torch.cat([diff_x1, diff_y2], dim=-1)) # B, L, topk, H*D
                p_br = self.W_br(torch.cat([diff_x2, diff_y2], dim=-1)) # B, L, topk, H*D

                p = p_tl + p_tr + p_bl + p_br
            else:
                p = None

            return p, topk_index
        else:
            return None, None


class VisualTokenExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = detectron2.config.get_cfg()
        self.backbone = VisionBackbone(config)
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        self.register_buffer(
            "pixel_mean",
            torch.Tensor(self.cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1),
        )
        self.register_buffer("pixel_std", torch.Tensor(self.cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1))

        self.scale = 0.25
        self.pool = RoiFeatExtraxtor(self.scale)

    def forward(self, images, line_bboxes):
        if isinstance(images, torch.Tensor):
            images_input = (images - self.pixel_mean) / self.pixel_std
        else:
            images_input = (images.tensor - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        features = features[0]
        features = self.pool(features, line_bboxes)
        return features


class GraphDocSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.fast_qkv = config.fast_qkv
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if config.fast_qkv:
            self.qkv_linear = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.use_rel_2d = config.use_rel_2d
        if self.use_rel_2d:
            self.use_rel_emb = config.use_rel_emb
            self.local_atten = config.local_atten
            if self.use_rel_emb:
                self.rel_bbox_query = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_bbox_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # B, L, L, H, D
        x = x.view(*new_x_shape)
        return x.permute(0, 3, 1, 2, 4)

    def compute_qkv(self, hidden_states):
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1,) * (q.ndimension() - 1) + (-1,)
                q = q + self.q_bias.view(*_sz)
                v = v + self.v_bias.view(*_sz)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        return q, k, v

    def forward(
        self,
        hidden_states,
        rel_bbox_emb,
        rel_bbox_index, # B, L, topk
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)
                
        query_layer = query_layer / math.sqrt(self.attention_head_size)
        
        # [BSZ, NAT, L, L]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.use_rel_2d:
            if self.use_rel_emb:
                q_bbox = self.rel_bbox_query(q)
                query_bbox_layer = self.transpose_for_scores(q_bbox)
                # (B, L, topk, H*D) -> (B, H, L, topk, D)
                rel_bbox_emb = self.transpose_for_bbox_scores(rel_bbox_emb)
                query_bbox_layer = query_bbox_layer / math.sqrt(self.attention_head_size)
                # cal rel bbox attention score
                attention_bbox_scores = torch.einsum('bhid,bhijd->bhij', query_bbox_layer, rel_bbox_emb)
                attention_scores = attention_scores.scatter_add(-1, rel_bbox_index.unsqueeze(1).expand_as(attention_bbox_scores), attention_bbox_scores)
            if self.local_atten:
                local_attention_mask = torch.ones_like(attention_scores)
                B, L, Topk = rel_bbox_index.shape
                local_attention_mask = local_attention_mask.float().scatter(-1, rel_bbox_index.unsqueeze(1).expand(B, self.num_attention_heads, L, Topk), 0.)
                attention_scores = attention_scores.float().masked_fill_(local_attention_mask.to(torch.bool), float(-1e8)) # remove too far token

        if self.has_relative_attention_bias:
            attention_scores += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos

        attention_scores = attention_scores.float().masked_fill_(attention_mask.to(torch.bool), float(-1e8)) # remove padding token
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class GraphDocAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = GraphDocSelfAttention(config)
        self.output = LayoutLMv2SelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        rel_bbox_emb,
        rel_bbox_index,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_outputs = self.self(
            hidden_states,
            rel_bbox_emb,
            rel_bbox_index,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class GraphDocLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = GraphDocAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = LayoutLMv2Attention(config)
        self.intermediate = LayoutLMv2Intermediate(config)
        self.output = LayoutLMv2Output(config)

    def forward(
        self,
        hidden_states,
        rel_bbox_emb,
        rel_bbox_index,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            rel_bbox_emb,
            rel_bbox_index,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class GLULayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(int(config.hidden_size * 2), config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_state, visual_emb):
        prob_z = self.transform(torch.cat((hidden_state, visual_emb), dim=-1))
        hidden_state = (1-prob_z)*hidden_state + prob_z*visual_emb
        return hidden_state


class GraphDocEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.textual_atten = LayoutLMv2Layer(config)
        self.use_visual_input = config.use_visual_input
        if self.use_visual_input:
            self.visual_atten = LayoutLMv2Layer(config)
            self.glulayer = nn.ModuleList([GLULayer(config) for _ in range(config.num_glu_layers)])
            self.num_glu_layers = config.num_glu_layers
        self.layer = nn.ModuleList([GraphDocLayer(config) for _ in range(config.num_hidden_layers)])

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.Linear(self.rel_pos_onehot_size, config.num_attention_heads, bias=False)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).type_as(hidden_states)
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
        self,
        textual_emb,
        visual_emb,
        rel_bbox_emb,
        rel_bbox_index,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        bbox=None,
        position_ids=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        rel_pos = self._cal_1d_pos_emb(textual_emb, position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(textual_emb, bbox) if self.has_spatial_attention_bias else None

        textual_emb = self.textual_atten(textual_emb, attention_mask, rel_pos=rel_pos, rel_2d_pos=rel_2d_pos)[0] # global self atten
        if self.use_visual_input:
            visual_emb = self.visual_atten(visual_emb, attention_mask, rel_pos=rel_pos, rel_2d_pos=rel_2d_pos)[0] # global self atten

        hidden_states = textual_emb
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.use_visual_input:
                hidden_states = self.glulayer[i](hidden_states, visual_emb) if i < self.num_glu_layers else hidden_states

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    rel_bbox_emb,
                    rel_bbox_index,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    rel_bbox_emb,
                    rel_bbox_index,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class GraphDocModel(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super(GraphDocModel, self).__init__(config)
        self.config = config
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.embeddings = GraphDocEmbeddings(config)

        self.use_visual_input = config.use_visual_input
        if self.use_visual_input:
            self.visual = VisualTokenExtractor(config)
            self.visual_proj = nn.Linear(config.vision_size, config.hidden_size)
            if self.has_visual_segment_embedding:
                self.visual_segment_embedding = nn.Parameter(nn.Embedding(1, config.hidden_size).weight[0])
            self.visual_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.visual_dropout = nn.Dropout(config.vision_hidden_dropout_prob)

        self.encoder = GraphDocEncoder(config)
        self.pooler = GraphDocPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    def _calc_text_embeddings(self, input_ids, input_ids_masks, input_embeds, bbox):
        if input_embeds is None:
            sentence_embeddings = self.embeddings.sentence_embeddings(input_ids, input_ids_masks)
        else:
            sentence_embeddings = input_embeds
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)
        if spatial_position_embeddings is not None:
            embeddings = sentence_embeddings + spatial_position_embeddings
        else:
            embeddings = sentence_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox):
        if self.use_visual_input:
            visual_embeddings = self.visual_proj(self.visual(image, bbox))
            spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)
            if spatial_position_embeddings is not None:
                embeddings = visual_embeddings + spatial_position_embeddings
            else:
                embeddings = visual_embeddings
            embeddings = self.visual_LayerNorm(embeddings)
            embeddings = self.visual_dropout(embeddings)
            return embeddings
        else:
            return None

    def forward(
        self,
        input_sentences=None,
        input_sentences_masks=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_sentences is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_sentences is not None:
            input_shape = input_sentences.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_sentences.device if input_sentences is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if bbox is None:
            bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

        text_layout_emb = self._calc_text_embeddings(input_ids=input_sentences, input_ids_masks=input_sentences_masks, input_embeds=inputs_embeds, bbox=bbox)
        rel_bbox_emb, rel_bbox_index = self.embeddings._cal_rel_position_embeddings(bbox=bbox, bbox_mask=attention_mask)
        visual_layout_emb = self._calc_img_embeddings(image=image, bbox=bbox)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            text_layout_emb,
            visual_layout_emb,
            rel_bbox_emb,
            rel_bbox_index,
            extended_attention_mask,
            bbox=bbox,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class GraphDocForEncode(LayoutLMv2PreTrainedModel):
    config_class = GraphDocConfig
    def __init__(self, config):
        super().__init__(config)
        self.layoutclm = GraphDocModel(config)
        self.init_weights()

    def forward(
        self,
        input_sentences=None,
        input_sentences_masks=None,
        bbox=None,
        image=None,
        unmask_image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mlm_masks=None,
        mvm_masks=None,
        unmask_embed=None,
        lcl_labels=None,
        dtc_labels=None,
        bdp_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutclm(
            input_sentences=input_sentences,
            input_sentences_masks=input_sentences_masks,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

class GraphDocForPretrain(LayoutLMv2PreTrainedModel):
    config_class = GraphDocConfig
    def __init__(self, config):
        super().__init__(config)
        self.layoutclm = GraphDocModel(config)
        self.hidden_size = config.hidden_size
        self.sequence_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.use_dtc = config.use_dtc
        if self.use_dtc:
            self.dtc_alpha = config.dtc_alpha
            self.dtc_head = nn.Linear(config.hidden_size, config.dtc_num)

        self.use_mlm = config.use_mlm
        if self.use_mlm:
            self.mlm_alpha = config.mlm_alpha
            self.mlm_head = nn.Linear(config.hidden_size, config.hidden_size)

        self.use_lcl = config.use_lcl
        if self.use_lcl:
            self.lcl_alpha = config.lcl_alpha
            self.lcl_head = nn.Linear(config.hidden_size, config.hidden_size)

        self.use_bdp = config.use_bdp
        if self.use_bdp:
            self.bdp_alpha = config.bdp_alpha
            self.bdp_blocks = config.bdp_blocks
            self.bdp_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.bdp_blocks * config.bdp_blocks)
            )

        self.use_mvm = config.use_mvm
        if self.use_mvm:
            self.mvm_alpha = config.mvm_alpha
            self.vision_size = config.vision_size
            self.mvm_head = nn.Linear(config.hidden_size, config.vision_size)

        self.init_weights()

    def forward(
        self,
        input_sentences=None,
        input_sentences_masks=None,
        bbox=None,
        image=None,
        unmask_image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mlm_masks=None,
        mvm_masks=None,
        unmask_embed=None,
        lcl_labels=None,
        dtc_labels=None,
        bdp_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutclm(
            input_sentences=input_sentences,
            input_sentences_masks=input_sentences_masks,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        batch_size = unmask_embed.size(0) # B
        seq_length = unmask_embed.size(1) # L
        sequence_output, pooler_output = outputs[0][:, :seq_length], outputs[1]
        sequence_output = self.sequence_dropout(sequence_output)

        # document type classification
        if self.use_dtc:
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            dtc_logits = self.dtc_head(pooler_output) # shape is (B, N), N is the number of document types
            dtc_loss = loss_fct(dtc_logits, dtc_labels)
            dtc_loss = self.dtc_alpha * dtc_loss.sum() / ((dtc_labels != -100).sum() + 1e-5)
        else:
            dtc_loss = 0.0
            dtc_logits = torch.zeros((batch_size, 15), dtype=sequence_output.dtype, device=sequence_output.device)

        # masked language model task
        if self.use_mlm:
            mlm_logits = self.mlm_head(sequence_output) # shape is (B, L, D)
            mlm_loss = F.smooth_l1_loss(mlm_logits, unmask_embed, reduction='none').mean(-1)
            mlm_loss = self.mlm_alpha * (mlm_loss * mlm_masks).sum() / (mlm_masks.sum() + 1e-5)
        else:
            mlm_loss = 0.0

        # language contrastive learning task
        if self.use_lcl:
            lcl_logits = self.lcl_head(sequence_output) # shape is (B, L, D)
            lcl_logits = torch.matmul(lcl_logits, unmask_embed.transpose(-1, -2)) # shape is (B, L, L)
            lcl_logits = lcl_logits.float().masked_fill_((1 - attention_mask[:, None,:]).to(torch.bool), float(-1e8))

            cal_lcl_acc_logits = []
            for logits, masks in zip(lcl_logits, mlm_masks):
                cal_lcl_acc_logits.append(logits[masks.to(torch.bool)])
            cal_lcl_acc_logits = align_logits(cal_lcl_acc_logits)

            active_masks = mlm_masks.to(torch.bool).view(-1)
            active_logits = lcl_logits.view(-1, lcl_logits.shape[-1])[active_masks] #shape is (N, L)
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            lcl_loss = loss_fct(active_logits, lcl_labels.view(-1)[(lcl_labels != -100).view(-1)])
            lcl_loss = self.lcl_alpha * lcl_loss.mean()

        else:
            lcl_loss = 0.0
            cal_lcl_acc_logits = torch.zeros((batch_size, seq_length, 15), dtype=sequence_output.dtype, device=sequence_output.device)
        
        # box direction prediction
        if self.use_bdp:
            # bdp_logits= self.bdp_head(sequence_output) # shape is (B, L, Num_Blocks)
            bdp_logits = self.bdp_head(self.layoutclm.embeddings._cal_spatial_position_embeddings(bbox)) # shape is (B, L, Num_Blocks)
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            bdp_loss = loss_fct(bdp_logits.view(-1, self.bdp_blocks * self.bdp_blocks), bdp_labels.view(-1))
            bdp_loss = self.bdp_alpha * bdp_loss.sum() / ((bdp_labels != -100).sum() + 1e-5)
        else:
            bdp_loss = 0.0
            bdp_logits = torch.zeros((batch_size, seq_length, 8), dtype=sequence_output.dtype, device=sequence_output.device)

        # masked vision model task
        if self.use_mvm:
            mvm_logits = self.mvm_head(sequence_output) # shape is (B, L, D)
            with torch.no_grad():
                mvm_labels = self.layoutclm.visual(unmask_image, bbox) # shape is (B, L, D)
            mvm_loss = F.smooth_l1_loss(mvm_logits, mvm_labels, reduction='none').mean(-1)
            mvm_loss = self.mvm_alpha * (mvm_loss * mvm_masks).sum() / (mlm_masks.sum() + 1e-5)
        else:
            mvm_loss = 0.0

        counter.update(dict(dtc_loss=dtc_loss, mlm_loss=mlm_loss, mvm_loss=mvm_loss, lcl_loss=lcl_loss, bdp_loss=bdp_loss))

        loss =  dtc_loss + mlm_loss + mvm_loss + lcl_loss + bdp_loss

        if torch.isnan(loss).any():
            logger.warning("nan is happend in loss, now loss is set to 0.0")
            loss = torch.zeros_like(loss).requires_grad_()

        if not return_dict:
            output = (dtc_logits.argmax(-1), bdp_logits.argmax(-1).view(batch_size, -1)) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GraphDocForTokenClassification(LayoutLMv2PreTrainedModel):
    config_class = GraphDocConfig
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutclm = GraphDocModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def get_input_embeddings(self):
        return self.layoutclm.embeddings.word_embeddings

    def forward(
        self,
        input_sentences=None,
        input_sentences_masks=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        items_polys_idxes=None,
        image_infos=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.layoutclm(
            input_sentences=input_sentences,
            input_sentences_masks=input_sentences_masks,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
            
        seq_length = inputs_embeds.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = 0.0

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GraphDocForClassification(LayoutLMv2PreTrainedModel):
    config_class = GraphDocConfig
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutclm = GraphDocModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def get_input_embeddings(self):
        return self.layoutclm.embeddings.word_embeddings

    def forward(
        self,
        input_sentences=None,
        input_sentences_masks=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.layoutclm(
            input_sentences=input_sentences,
            input_sentences_masks=input_sentences_masks,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
            
        pooler_output = outputs[1]
        pooler_output = self.dropout(pooler_output)

        # document type classification
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        logits = self.classifier(pooler_output) # shape is (B, N), N is the number of document types
        loss = loss_fct(logits, labels)
        loss = loss.sum() / ((labels != -100).sum() + 1e-5)

        if not return_dict:
            output = (logits.argmax(-1), ) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )