import re
import time
from tqdm import tqdm
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from preprocess.preprocess import TextPreprocessor


class KamusSlangTemplate:
    def __init__(self, texts) -> None:
        self.texts = texts
        self._template = pd.DataFrame(
            columns=["Slang", "Makna", "No_Konteks", "Konteks"])

    @property
    def template(self):
        return self._template

    def _prepare_text(self, text) -> dict:
        preprocessor = TextPreprocessor()
        text = preprocessor.clean(text)
        tokens = preprocessor.tokenize(text)
        filtered_tokens = preprocessor.filter(tokens)
        stemmed_tokens = preprocessor.stem(filtered_tokens)
        return {
            "filtered": filtered_tokens,
            "stemmed": stemmed_tokens,
        }

    def create(self):
        slangs = []
        contexts = []
        indices = []
        factory = StemmerFactory()
        kata_dasar_list = factory.get_words()
        kata_dasar_list.remove("")
        for i, text in enumerate(tqdm(self.texts)):
            text = re.sub(r"\s+", " ", text)
            tokens = self._prepare_text(text)
            for j, token in enumerate(tokens["stemmed"]):
                if token not in kata_dasar_list:
                    if tokens["filtered"][j] not in slangs:
                        slangs.append(tokens["filtered"][j])
                        contexts.append(text)
                        indices.append(i)

            time.sleep(0.001)
        self._template["Slang"] = slangs
        self._template["Konteks"] = contexts
        self._template["No_Konteks"] = indices