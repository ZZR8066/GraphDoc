from libs.utils.vocab import FunsdTokenTypeVocab

train_lrc_paths = ["/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/funsd/dataset/extract_word_lrc/train/infos.lrc"]
train_image_dirs = ["/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/funsd/dataset/training_data/images/"]

valid_lrc_paths = ["/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/funsd/dataset/extract_word_lrc/test/infos.lrc"]
valid_image_dirs = ["/yrfs1/intern/zrzhang6/DocumentPretrain/dataset/funsd/dataset/testing_data/images/"]

funsd_vocab = FunsdTokenTypeVocab()

max_length = 512