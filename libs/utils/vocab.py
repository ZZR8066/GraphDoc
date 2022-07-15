import numpy as np


class DocTypeVocab:
    key_words = [
    'letter', 'form', 'email', 'handwritten', 'advertisement', 'scientific report', \
        'scientific publication', 'specification', 'file folder', 'news article', \
            'budget', 'invoice', 'presentation', 'questionnaire', 'resume', 'memo', 'docbank' ]

    def __init__(self):
        self._words_ids_map = dict()
        self._ids_words_map = dict()

        for word_id, word in enumerate(self.key_words):
            self._words_ids_map[word] = word_id
            self._ids_words_map[word_id] = word
    
    def __len__(self):
        return len(self._words_ids_map)

    def word_to_id(self, word):
        return self._words_ids_map[word]

    def words_to_ids(self, words):
        return [self.word_to_id(word) for word in words]

    def id_to_word(self, word_id):
        return self._ids_words_map[word_id]
    
    def ids_to_words(self, words_id):
        return [self.id_to_word(word_id) for word_id in words_id]


class FunsdTokenTypeVocab(DocTypeVocab):
    key_words = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]


class FunsdEntityVocab(DocTypeVocab):
    key_words = ["OTHER", "HEADER", "QUESTION", "ANSWER"]

class HuaweikieEntityVocab(DocTypeVocab):
    key_words = ["DATE", "COMPANY", "TAX", "TOTAL", "NAME", "CNT", "PRICE", "OTHER", "PRICE&CNT", "CNT&NAME"]

class CordEntityVocab(DocTypeVocab):
    key_words = ['OTHER', 'MENU_CNT', 'MENU_UNITPRICE', 'MENU_NM', 'MENU_NUM', 'MENU_PRICE', 'MENU_DISCOUNTPRICE', \
        'MENU_ITEMSUBTOTAL', 'MENU_ETC', 'MENU_SUB_CNT', 'MENU_SUB_ETC', 'MENU_SUB_NM', 'MENU_SUB_PRICE', 'MENU_SUB_UNITPRICE', \
            'MENU_VATYN', 'SUB_TOTAL_DISCOUNT_PRICE', 'SUB_TOTAL_ETC', 'SUB_TOTAL_OTHERSVC_PRICE', 'SUB_TOTAL_SERVICE_PRICE', 'SUB_TOTAL_SUBTOTAL_PRICE', \
                'SUB_TOTAL_TAX_PRICE', 'TOTAL_CASHPRICE', 'TOTAL_CHANGEPRICE', 'TOTAL_CREDITCARDPRICE', 'TOTAL_EMONEYPRICE', 'TOTAL_MENUQTY_CNT', 'TOTAL_MENUTYPE_CNT', \
                    'TOTAL_TOTAL_ETC', 'TOTAL_TOTAL_PRICE', 'VOID_MENU_NM', 'VOID_MENU_PRICE']


class SroieEntityVocab(DocTypeVocab):
    key_words = ['O', 'COMPANY', 'ADDRESS', 'DATE', 'TOTAL']