from libs.utils.vocab import DocTypeVocab
from libs.utils.counter import Counter


train_lrc_paths = [\
    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/extract_lrc/test/infos.lrc", \
        "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/extract_lrc/train/infos.lrc", \
            "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/extract_lrc/val/infos.lrc", \
                "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/info_lrc/test/infos.lrc", \
                    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/info_lrc/train/infos.lrc", \
                        "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/info_lrc/val/infos.lrc"
    ]

train_sorted_lrc_paths = [\
    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/extract_sorted_lrc/test/infos.lrc", \
        "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/extract_sorted_lrc/train/infos.lrc", \
            "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/extract_sorted_lrc/val/infos.lrc", \
                "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/extract_sorted_lrc/test/infos.lrc", \
                    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/extract_sorted_lrc/train/infos.lrc", \
                        "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/extract_sorted_lrc/val/infos.lrc"
    ]

train_image_dirs = [\
    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/rvl-cdip/images/", \
        "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/rvl-cdip/images/", \
            "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/rvl-cdip/images/", \
                "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/DocBank_500K_images/", \
                    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/DocBank_500K_images/", \
                        "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/DocBank_500K_images/"
    ]

train_rvlcdip_lrc_paths = [\
    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/extract_sorted_lrc/test/infos.lrc", \
        "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/extract_sorted_lrc/train/infos.lrc", \
            "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/extract_sorted_lrc/val/infos.lrc",
    ]

train_rvlcdip_image_dirs = [\
    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/rvl-cdip/images/", \
        "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/rvl-cdip/images/", \
            "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/rvl-cdip/images/", \
    ]

train_docbank_lrc_paths =[\
    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/info_lrc/test/infos.lrc", \
        "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/info_lrc/train/infos.lrc", \
            "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/info_lrc/val/infos.lrc"
    ]

train_docbank_image_dirs = [\
    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/DocBank_500K_images/", \
        "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/DocBank_500K_images/", \
            "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/DocBank_500K_images/"

    ]

train_ucfs_180w_lrc_paths = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/extract_lrc/180w/train/infos.lrc"
    ]

train_ucfs_180w_image_dirs = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/pdf/images"
    ]

train_ucfs_700w_lrc_paths = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/extract_lrc/700w/train/infos.lrc"
    ]

train_ucfs_700w_image_dirs = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/pdf/images"
    ]

train_ucfs_180w_word_lrc_paths = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/extract_lrc/180w/train/word_level/infos.lrc"
    ]

train_ucfs_180w_word_image_dirs = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/pdf/images"
    ]

train_ucfs_700w_word_lrc_paths = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/extract_lrc/700w/train/word_level/infos.lrc"
    ]

train_ucfs_700w_word_image_dirs = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/pdf/images"
    ]

train_ucfs_700w_word_v2_lrc_paths = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/extract_lrc/700w/train/word_level_v2/infos.lrc"
    ]

train_ucfs_700w_word_v2_image_dirs = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/pdf/images"
    ]

train_ucfs_900w_word_lrc_paths = [\
    "/yrfs2/cv1/jszhang6/zrzhang6/DocumentPretrain/dataset/ucfs/extract_lrc/900w/train/word_level/infos.lrc"
    ]

train_ucfs_900w_word_image_dirs = [\
    "/yrfs2/cv1/jszhang6/zrzhang6/DocumentPretrain/dataset/ucfs/images/images"
    ]

# valid path
valid_lrc_paths = [\
    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/extract_lrc/small_test/infos.lrc",\
        "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/info_lrc/small_test/infos.lrc"
    ]

valid_sorted_lrc_paths = [\
    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/extract_sorted_lrc/small_test/infos.lrc",\
        "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/extract_sorted_lrc/small_test/infos.lrc"
    ]

valid_image_dirs = [\
    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/rvl-cdip/images/", \
        "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/DocBank_500K_images/"
    ]

valid_rvlcdip_lrc_paths = [\
    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/extract_sorted_lrc/small_test/infos.lrc",
    ]

valid_rvlcdip_image_dirs=[\
    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/RVL_CDIP/rvl-cdip/images/"
    ]

valid_docbank_lrc_paths = [\
    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/extract_sorted_lrc/small_test/infos.lrc"
    ]

valid_docbank_image_dirs = [\
    "/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/DocBank/DocBank_500K_images/"
    ]

valid_ucfs_180w_lrc_paths = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/extract_lrc/180w/test/infos.lrc"
    ]

valid_ucfs_180w_image_dirs = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/pdf/images"
    ]

valid_ucfs_700w_lrc_paths = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/extract_lrc/700w/test/infos.lrc"
    ]

valid_ucfs_700w_image_dirs = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/pdf/images"
    ]

valid_ucfs_180w_word_lrc_paths = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/extract_lrc/180w/test/word_level/infos.lrc"
    ]

valid_ucfs_180w_word_image_dirs = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/pdf/images"
    ]

valid_ucfs_700w_word_lrc_paths = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/extract_lrc/700w/test/word_level/infos.lrc"
    ]

valid_ucfs_700w_word_image_dirs = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/pdf/images"
    ]

valid_ucfs_700w_word_v2_lrc_paths = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/extract_lrc/700w/test/word_level_v2/infos.lrc"
    ]

valid_ucfs_700w_word_v2_image_dirs = [\
    "/yrfs1/hyperbrain/hyyang11/zrzhang6/DocPretrain/dataset/ucfs/pdf/images"
    ]

valid_ucfs_900w_word_lrc_paths = [\
    "/yrfs2/cv1/jszhang6/zrzhang6/DocumentPretrain/dataset/ucfs/extract_lrc/900w/test/word_level/infos.lrc"
    ]

valid_ucfs_900w_word_image_dirs = [\
    "/yrfs2/cv1/jszhang6/zrzhang6/DocumentPretrain/dataset/ucfs/images/images"
    ]


# extract by easyocr and tokenize by robert-sentence-bert
train_rvlcdip_sentence_lrc_paths = ["/work1/cv1/jszhang6/DocumentPretrain/dataset/rvlcdip/extract_lrc/line/v2/extract_lrc/infos.lrc"]
# train_rvlcdip_sentence_lrc_paths = ["/work1/cv1/jszhang6/DocumentPretrain/dataset/rvlcdip/extract_lrc/line/v1/extract_lrc/small_val.lrc"]
train_rvlcdip_sentence_image_dirs = [None]

valid_rvlcdip_sentence_lrc_paths = ["/work1/cv1/jszhang6/DocumentPretrain/dataset/rvlcdip/extract_lrc/line/v1/extract_lrc/small_val.lrc"]
valid_rvlcdip_sentence_image_dirs = [None]

train_ucfs_sentence_200w_lrc_paths = ["/work1/cv1/jszhang6/DocumentPretrain/dataset/ucfs/word/v1/extract_lrc/infos.lrc"]
train_ucfs_sentence_200w_image_dirs = [None]

valid_ucfs_sentence_200w_lrc_paths = ["/work1/cv1/jszhang6/DocumentPretrain/dataset/ucfs/word/v1/extract_lrc/infos.lrc"]
valid_ucfs_sentence_200w_image_dirs = [None]

# dataset path dict
datasets = dict(
    # train
    train_lrc_paths=train_lrc_paths,
    train_sorted_lrc_paths=train_sorted_lrc_paths,
    train_image_dirs=train_image_dirs,
    train_rvlcdip_lrc_paths=train_rvlcdip_lrc_paths,
    train_rvlcdip_image_dirs=train_rvlcdip_image_dirs,
    train_docbank_lrc_paths=train_docbank_lrc_paths,
    train_docbank_image_dirs=train_docbank_image_dirs,
    train_ucfs_180w_lrc_paths=train_ucfs_180w_lrc_paths,
    train_ucfs_180w_image_dirs=train_ucfs_180w_image_dirs,
    train_ucfs_700w_lrc_paths=train_ucfs_700w_lrc_paths,
    train_ucfs_700w_image_dirs=train_ucfs_700w_image_dirs,
    train_ucfs_180w_word_lrc_paths=train_ucfs_180w_word_lrc_paths,
    train_ucfs_180w_word_image_dirs=train_ucfs_180w_word_image_dirs,
    train_ucfs_700w_word_lrc_paths=train_ucfs_700w_word_lrc_paths,
    train_ucfs_700w_word_image_dirs=train_ucfs_700w_word_image_dirs,
    train_ucfs_700w_word_v2_lrc_paths=train_ucfs_700w_word_v2_lrc_paths,
    train_ucfs_700w_word_v2_image_dirs=train_ucfs_700w_word_v2_image_dirs,
    train_ucfs_900w_word_lrc_paths=train_ucfs_900w_word_lrc_paths,
    train_ucfs_900w_word_image_dirs=train_ucfs_900w_word_image_dirs,

    # valid
    valid_lrc_paths=valid_lrc_paths,
    valid_sorted_lrc_paths=valid_sorted_lrc_paths,
    valid_image_dirs=valid_image_dirs,
    valid_rvlcdip_lrc_paths=valid_rvlcdip_lrc_paths,
    valid_rvlcdip_image_dirs=valid_rvlcdip_image_dirs,
    valid_docbank_lrc_paths=valid_docbank_lrc_paths,
    valid_docbank_image_dirs=valid_docbank_image_dirs,
    valid_ucfs_180w_lrc_paths=valid_ucfs_180w_lrc_paths,
    valid_ucfs_180w_image_dirs=valid_ucfs_180w_image_dirs,
    valid_ucfs_700w_lrc_paths=valid_ucfs_700w_lrc_paths,
    valid_ucfs_700w_image_dirs=valid_ucfs_700w_image_dirs,
    valid_ucfs_180w_word_lrc_paths=valid_ucfs_180w_word_lrc_paths,
    valid_ucfs_180w_word_image_dirs=valid_ucfs_180w_word_image_dirs,
    valid_ucfs_700w_word_lrc_paths=valid_ucfs_700w_word_lrc_paths,
    valid_ucfs_700w_word_image_dirs=valid_ucfs_700w_word_image_dirs,
    valid_ucfs_700w_word_v2_lrc_paths=valid_ucfs_700w_word_v2_lrc_paths,
    valid_ucfs_700w_word_v2_image_dirs=valid_ucfs_700w_word_v2_image_dirs,
    valid_ucfs_900w_word_lrc_paths=valid_ucfs_900w_word_lrc_paths,
    valid_ucfs_900w_word_image_dirs=valid_ucfs_900w_word_image_dirs,

    # sentence level
    train_rvlcdip_sentence_lrc_paths=train_rvlcdip_sentence_lrc_paths,
    train_rvlcdip_sentence_image_dirs=train_rvlcdip_sentence_image_dirs,
    valid_rvlcdip_sentence_lrc_paths=valid_rvlcdip_sentence_lrc_paths,
    valid_rvlcdip_sentence_image_dirs=valid_rvlcdip_sentence_image_dirs,
    
    train_ucfs_sentence_200w_lrc_paths=train_ucfs_sentence_200w_lrc_paths,
    train_ucfs_sentence_200w_image_dirs=train_ucfs_sentence_200w_image_dirs,
    valid_ucfs_sentence_200w_lrc_paths=valid_ucfs_sentence_200w_lrc_paths,
    valid_ucfs_sentence_200w_image_dirs=valid_ucfs_sentence_200w_image_dirs
)

# document type vocab
doctype_vocab = DocTypeVocab()

# counter for show each item loss
counter = Counter(cache_nums=1000)