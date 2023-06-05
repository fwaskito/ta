# Created Date: Wed, May 24th 2023
# Author: F. Waskito
# Last Modified: Sat, Jun 3rd 2023 1:51:51 PM

from time import sleep
from tqdm import tqdm
from pandas import DataFrame
from preprocess.preprocessing import TextPreprocessor


class KamusAntonimTemplate:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self._preperator = TextPreprocessor()
        self._stopwords_id = self._preperator.get_stopwords()
        self._template = DataFrame(columns=[
            "Kata",
            "Antonim",
            "No_Konteks",
            "Konteks",
        ])

    @property
    def template(self) -> DataFrame:
        return self._template

    def _prepare_text(self, text) -> list[str]:
        text = self._preperator.clean(text)
        text = self._preperator.standardize(text)
        return self._preperator.tokenize(text)

    def _find_negation_word(self, i, tokens) -> None:
        j, n_token = 0, len(tokens)
        while j < n_token:
            token = tokens[j]
            if token in "tidak" and j + 1 < n_token:
                neg_word = tokens[j + 1]
                if neg_word not in self._stopwords_id:
                    neg_words = self.template["Kata"].to_list()
                    neg_word = self._preperator.stem(neg_word)
                    if neg_word not in neg_words:
                        index = len(self._template)
                        context = " ".join(tokens)
                        self._template.loc[index, "Kata"] = neg_word
                        self._template.loc[index, "Konteks"] = context
                        self._template.loc[index, "No_Konteks"] = i
                        j += 1

                    j += 1
                    continue
            j += 1

    def create(self) -> None:
        for i, text in enumerate(tqdm(self._texts)):
            tokens = self._prepare_text(text)
            self._find_negation_word(i, tokens)
            sleep(0.001)