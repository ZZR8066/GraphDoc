# coding=utf-8
from transformers.utils import logging
from ..layoutlmv2 import LayoutLMv2Config


logger = logging.get_logger(__name__)


class GraphDocConfig(LayoutLMv2Config):
    model_type = "graphdoc"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_glu_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        vision_hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        max_2d_position_embeddings=1024,
        max_rel_pos=128,
        rel_pos_bins=32,
        fast_qkv=True,
        max_rel_2d_pos=256,
        rel_2d_pos_bins=64,
        rel_topk=56,
        convert_sync_batchnorm=True,
        image_feature_pool_shape=[7, 7, 256],
        coordinate_size=128,
        shape_size=128,
        has_relative_attention_bias=True,
        has_spatial_attention_bias=True,
        has_visual_segment_embedding=False,
        use_dtc=False,
        dtc_alpha=1,
        dtc_num=16,
        use_mlm=True,
        mlm_alpha=1,
        mlm_prob=0.15,
        is_cover=False,
        use_mvm=False,
        mvm_alpha=1,
        mvm_prob=0.075,
        use_lcl=False,
        lcl_alpha=1,
        use_bdp=False,
        bdp_alpha=1,
        bdp_blocks=4,
        pos_embed_size=24,
        expand_wh_scale=5.0,
        use_visual_input=True,
        use_abs_emb=True,
        use_rel_2d=False,
        use_rel_emb=True,
        local_atten=False,
        abs_emb_type='Liner',
        datasets=['docbank', 'rvlcdip'],
        sentence_model='/yrfs2/cv1/jszhang6/zrzhang6/PretrainModel/Transformer/sup-simcse-bert-base-uncased/',
        mask_embed='/yrfs2/cv1/jszhang6/zrzhang6/DocumentPretrain/model/PretrainLM/libs/configs/layoutclmV28/mask_embedding.npy',
        vision_freeze=False,
        backbone_cfg={'attn_drop_rate': 0.0, 'depths': [2, 2, 6, 2], 'drop_path_rate': 0.2, 
                    'drop_rate': 0.0, 'embed_dims': 96, 'mlp_ratio': 4, 'num_heads': [3, 6, 12, 24], 
                    'out_indices': (0, 1, 2, 3), 'patch_norm': True, 'qk_scale': None, 'qkv_bias': True, 
                    'type': 'SwinTransformer', 'window_size': 7, 'with_cp': False},
        neck_cfg={'in_channels': [96, 192, 384, 768], 'num_outs': 5, 'out_channels': 256, 'type': 'FPN'},
        vision_pretrain='/work1/cv1/jszhang6/TSR/code/mmdetection/experiments/publaynet/centernet_swin_t/epoch_12.pth',
        vision_size=256,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs,
        )
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.max_rel_pos = max_rel_pos
        self.rel_pos_bins = rel_pos_bins
        self.fast_qkv = fast_qkv
        self.max_rel_2d_pos = max_rel_2d_pos
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.convert_sync_batchnorm = convert_sync_batchnorm
        self.image_feature_pool_shape = image_feature_pool_shape
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.has_relative_attention_bias = has_relative_attention_bias
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.has_visual_segment_embedding = has_visual_segment_embedding

        self.datasets = datasets
        self.vision_hidden_dropout_prob = vision_hidden_dropout_prob
        self.num_glu_layers = num_glu_layers
        self.pos_embed_size = pos_embed_size
        self.expand_wh_scale = expand_wh_scale
        self.use_rel_2d = use_rel_2d
        self.use_rel_emb = use_rel_emb
        self.local_atten = local_atten
        self.abs_emb_type = abs_emb_type
        self.rel_topk = rel_topk
        self.mlm_prob = mlm_prob
        self.mlm_alpha = mlm_alpha
        self.sentence_model = sentence_model
        self.mask_embed = mask_embed
        self.vision_freeze = vision_freeze
        self.backbone_cfg = backbone_cfg
        self.neck_cfg = neck_cfg
        self.vision_pretrain = vision_pretrain
        self.vision_size = vision_size
        self.use_mlm = use_mlm
        self.use_lcl = use_lcl
        self.lcl_alpha = lcl_alpha
        self.use_dtc = use_dtc
        self.dtc_alpha = dtc_alpha
        self.dtc_num = dtc_num
        self.use_bdp = use_bdp
        self.bdp_alpha = bdp_alpha
        self.bdp_blocks = bdp_blocks
        self.use_abs_emb = use_abs_emb
        self.use_mvm = use_mvm
        self.is_cover = is_cover
        self.mvm_alpha = mvm_alpha
        self.mvm_prob = mvm_prob
        self.use_visual_input = use_visual_input