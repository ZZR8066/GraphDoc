# coding=utf-8
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.utils import logging

from ..layoutlmv2 import LayoutLMv2ForRelationExtraction, LayoutLMv2ForTokenClassification, LayoutLMv2Model
from .configuration_layoutxlm import LayoutXLMConfig
from transformers.modeling_outputs import TokenClassifierOutput

logger = logging.get_logger(__name__)

LAYOUTXLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "layoutxlm-base",
    "layoutxlm-large",
]


class LayoutXLMForPretrain(LayoutLMv2ForTokenClassification):
    config_class = LayoutXLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.num_tokens = config.num_tokens
        self.mvlm_cls = nn.Linear(config.hidden_size, config.num_tokens)
        self.tia_cls = nn.Linear(config.hidden_size, 2)
        self.tim_cls = nn.Linear(config.hidden_size, 2)

        total_alpha = config.mvlm_alpha + config.tia_alpha + config.tim_alpha
        self.mvlm_alpha = config.mvlm_alpha / total_alpha
        self.tia_alpha = config.tia_alpha / total_alpha
        self.tim_alpha = config.tim_alpha / total_alpha

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mvlm_labels=None,
        tia_labels=None,
        tim_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # with torch.no_grad():
        outputs = self.layoutlmv2(
            input_ids=input_ids,
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
            
        seq_length = input_ids.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output)

        loss = None
        mvlm_logits = None
        tia_logits = None
        tim_logits = None

        if mvlm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            mvlm_logits = self.mvlm_cls(sequence_output)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = mvlm_logits.view(-1, self.num_tokens)[active_loss]
                active_labels = mvlm_labels.view(-1)[active_loss]
                mvlm_loss = loss_fct(active_logits, active_labels)
            else:
                mvlm_loss = loss_fct(mvlm_logits.view(-1, self.num_tokens), mvlm_labels.view(-1))
            mvlm_loss = mvlm_loss.sum() / ((mvlm_labels != -100).sum() + 1e-5)
            if loss is not None:
                loss += self.mvlm_alpha * mvlm_loss
            else:
                loss = self.mvlm_alpha * mvlm_loss

        if tia_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            tia_logits = self.tia_cls(sequence_output)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = tia_logits.view(-1, 2)[active_loss]
                active_labels = tia_labels.view(-1)[active_loss]
                tia_loss = loss_fct(active_logits, active_labels)
            else:
                tia_loss = loss_fct(tia_logits.view(-1, 2), tia_labels.view(-1))
            tia_loss = tia_loss.sum() / ((tia_labels != -100).sum() + 1e-5)
            if loss is not None:
                loss += self.tia_alpha * tia_loss
            else:
                loss = self.tia_alpha * tia_loss

        if tim_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            tim_logits = self.tim_cls(sequence_output[:, 0])
            tim_loss = loss_fct(tim_logits.view(-1, 2), tim_labels.view(-1))
            tim_loss = tim_loss.sum() / ((tim_labels != -100).sum() + 1e-5)
            if loss is not None:
                loss += self.tim_alpha * tim_loss
            else:
                loss = self.tim_alpha * tim_loss

        if not return_dict:
            output = (mvlm_logits.argmax(-1), tia_logits.argmax(-1), tim_logits.argmax(-1)) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LayoutXLMModel(LayoutLMv2Model):
    config_class = LayoutXLMConfig


class LayoutXLMForTokenClassification(LayoutLMv2ForTokenClassification):
    config_class = LayoutXLMConfig

    
class LayoutXLMForRelationExtraction(LayoutLMv2ForRelationExtraction):
    config_class = LayoutXLMConfig