# Created Date: Sat, Mar 18th 2023
# Author: F. Waskito
# Last Modified: Fri, Jun 2nd 2023 11:36:35 PM

import re
from re import sub
from typing import Union
from string import punctuation
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collection.slang.dictionary import KamusSlang
from collection.antonym.dictionary import KamusAntonim


class TextPreprocessor:
    def __init__(self) -> None:
        self.stopwords_id: set = self.get_stopwords()
        self.kamus_slang: KamusSlang = KamusSlang()
        self.kamus_antonim: KamusAntonim =KamusAntonim()
        self.stemmer = StemmerFactory().create_stemmer()
        self.kamus_slang.generate()
        self.kamus_antonim.generate()

    def get_stopwords(self) -> set:
        stopwords_id = stopwords.words("indonesian")
        additions = [
            "ah", "ih", "uh", "eh", "oh", "hai", "halo",
            "oi", "ayo", "yuk", "ya", "yah", "mah", "nah",
            "wah", "alah", "oalah", "dah", "dih", "aduh",
            "deh", "loh", "kok", "kek", "entah", "sih",
            "si", "wow", "aw", "ea", "cie", "kak", "dik",
            "mba", "mas", "bang", "bu", "om", "bund",
            "nder", "thor", "nge", "kah", "ber", "an",
            "ku", "mu", "nya", "tawa", "berdeham"
        ]
        return set([*stopwords_id, *additions])

    def clean(self, text: str) -> str:
        cleaner = TextCleaner(text)
        cleaner.clean_all()
        return cleaner.text

    def standardize(self, text: str) -> str:
        tokens = text.split(" ")
        for i, token in enumerate(tokens):
            meaning = self.kamus_slang.get_meaning(token)
            tokens[i] = meaning
        return " ".join(tokens)

    def tokenize(self, text: str) -> list[str]:
        return text.split(" ")

    def filter(self, tokens: list[str]) -> list[str]:
        filtered_tokens = []
        for token in tokens:
            if token not in self.stopwords_id:
                filtered_tokens.append(token)
        return filtered_tokens

    def stem(self, tokens: list[str]) -> str:
        stemmed_tokens = []
        for token in tokens:
            stemmed_token = self.stemmer.stem(token)
            stemmed_tokens.append(stemmed_token)
        return " ".join(stemmed_tokens)

    def _handle_negation(self, token) -> Union[str, None]:
        if token not in self.stopwords_id:
            token = self.stemmer.stem(token)
            antonym = self.kamus_antonim.get_antonym(token)
            if antonym:
                if antonym not in self.stopwords_id:
                    return self.stemmer.stem(antonym)
        return None

    def filter_stem(self, tokens: list[str]) -> str:
        """
        Method that includes handling negations between the
        filtering and stemming processes.

        Before use this method, make sure the word "tidak"
        is removed from stopwords instance property.
        -----------------------------
        object = TextPreprocessor()
        object.stopwords_id.remove("tidak")

        """
        i, n_token = 0, len(tokens)
        stemmed_tokens = []
        while i < n_token:
            token = tokens[i]
            if token not in self.stopwords_id:
                if token == "tidak" and (i + 1) < n_token:
                    antonym = self._handle_negation(tokens[i + 1])
                    if antonym:
                        stemmed_tokens.append(antonym)
                        i += 1

                    i += 1
                    continue
                token = self.stemmer.stem(token)
                stemmed_tokens.append(token)
            i += 1
        return " ".join(stemmed_tokens)


class TextCleaner:
    def __init__(self, text=str) -> None:
        self.text = text.lower()

    # def remove_kaomoji(self):
    #     """Remove Japanese style emoji.
    #     Useless in Windows.
    #     """
    #     path = "data/dictionary/kaomoji.txt"
    #     with open(
    #             path + 'kaomoji',
    #             encoding='utf-8',
    #             mode = 'r',
    #     ) as file:
    #         kaomoji = file.read()
    #         kaomoji = kaomoji.rstrip("|")
    #         kaomoji = re.sub(r"\\", r"\\\\", kaomoji)
    #         kaomoji = re.sub(r"\(", "\(", kaomoji)
    #         kaomoji = re.sub(r"\)", "\)", kaomoji)
    #         self.text = re.sub(r""+kaomoji, "", self.text)

    def remove_emoji(self) -> None:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002500-\U00002BEF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+",
            flags=re.UNICODE,
        )
        self.text = emoji_pattern.sub(r"", self.text)

    def remove_html_entity(self) -> None:
        regex = r"&.*;"
        self.text = sub(regex, " ", self.text)

    def remove_mention(self) -> None:
        regex = r"@[^\s]+"
        self.text = sub(regex, "", self.text)

    def replace_url(self, chars='link') -> None:
        regex = r"http\S+"
        self.text = sub(regex, chars, self.text)

    def separate_plural(self) -> None:
        regex = "-"
        self.text = sub(regex, " ", self.text)

    def separate_or(self) -> None:
        regex = "/"
        self.text = sub(regex, " ", self.text)

    def remove_number(self) -> None:
        regex = r"\d+"
        self.text = sub(regex, "", self.text)

    def remove_period_comma(self) -> None:
        regex = r"\.|\,|…"
        self.text = sub(regex, " ", self.text)

    def remove_quotation(self) -> None:
        regex = '“|”|„|‟|″|‴|‶|‷|❝|❞|ʺ|˝'
        self.text = sub(regex, "", self.text)

    def remove_apostrophe(self) -> None:
        regex = "‘|’|‚|‛|′|‵|ʹ|ʻ|ʼ|ʽ|ʾ|ʿ|ˈ|ˊ|ˋ"
        self.text = sub(regex, "", self.text)

    def remove_hyphen_dash(self) -> None:
        regex = "‑|‒|–|—|―"
        self.text = sub(regex, "", self.text)

    def remove_math_symbol(self) -> None:
        regex = "≠|²"
        self.text = sub(regex, "", self.text)

    def remove_other_punct(self) -> None:
        self.text = self.text.translate(str.maketrans(
            "",
            "",
            punctuation,
        ))

    def remove_leading_space(self) -> None:
        regex = r"^\s+"
        self.text = sub(regex, "", self.text)

    def remove_trailing_space(self) -> None:
        regex = r"\s+$"
        self.text = sub(regex, "", self.text)

    def remove_multiple_space(self) -> None:
        regex = r"\s+"
        self.text = sub(regex, " ", self.text)

    def clean_all(self) -> None:
        self.remove_emoji()
        self.remove_html_entity()
        self.remove_mention()
        self.replace_url()
        self.separate_plural()
        self.separate_or()
        self.remove_number()
        self.remove_period_comma()
        self.remove_quotation()
        self.remove_apostrophe()
        self.remove_hyphen_dash()
        self.remove_math_symbol()
        self.remove_other_punct()
        self.remove_leading_space()
        self.remove_trailing_space()
        self.remove_multiple_space()