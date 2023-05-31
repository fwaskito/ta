import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collection.slang.convertion import SlangConverter


class TextPreprocessor:
    def __init__(self) -> None:
        self._stopwords_id = self.get_stopwords()
        self._stemmer = StemmerFactory().create_stemmer()
        self._slang_converter = SlangConverter()

    def get_stopwords(self) -> set:
        stopwords_id = stopwords.words("indonesian")
        additions = [
            "ah", "ih", "uh", "eh", "oh", "hai", "halo", "oi", "ayo",
            "yuk", "ya", "yah", "mah", "nah", "wah", "alah", "oalah",
            "dah", "dih", "aduh", "deh", "loh", "kok", "kek", "entah",
            "sih", "si", "wow", "aw", "ea", "cie", "kak", "dik", "mba",
            "mas", "bang", "bu", "om", "bund", "nder", "thor", "nge",
            "kah", "ber", "an", "ku", "mu", "nya", "tawa", "berdeham"
        ]
        return set([*stopwords_id, *additions])

    def clean(self, text) -> str:
        return TextCleaner(text, full=True).text

    def standardize(self, text) -> str:
        return self._slang_converter.convert(text)

    def tokenize(self, text) -> list[str]:
        return word_tokenize(text)

    def filter(self, tokens) -> list[str]:
        filtered_tokens = []
        for token in tokens:
            if token not in self._stopwords_id:
                filtered_tokens.append(token)
        return filtered_tokens

    def stem(self, tokens) -> str:
        if type(tokens) is list:
            stemmed_tokens = []
            for token in tokens:
                stemmed_token = self._stemmer.stem(token)
                stemmed_tokens.append(stemmed_token)

            return " ".join(stemmed_tokens)
        return self._stemmer.stem(tokens)


class TextCleaner:
    def __init__(self, text=str, full=False) -> None:
        self.text = text.lower()
        if full:
            self._clean_all()

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

    def remove_new_line(self) -> None:
        regex = r"\s+"
        self.text = re.sub(regex, " ", self.text)

    def remove_number(self) -> None:
        regex = r"\d+"
        self.text = re.sub(regex, "", self.text)

    def remove_mention(self) -> None:
        regex = r"@[^\s]+"
        self.text = re.sub(regex, "", self.text)

    def replace_url(self, chars='link') -> None:
        regex = r"http\S+"
        self.text = re.sub(regex, chars, self.text)

    def separate_plural(self) -> None:
        regex = r"-"
        self.text = re.sub(regex, " ", self.text)

    def replace_or(self, chars=" ") -> None:
        regex = "/"
        self.text = re.sub(regex, chars, self.text)

    def replace_and(self, chars="dan") -> None:
        regex = "&amp;"
        self.text = re.sub(regex, chars, self.text)

    def remove_period_comma(self) -> None:
        regex = r"\.|\,|…"
        self.text = re.sub(regex, " ", self.text)

    def remove_quotation(self) -> None:
        regex = '“|”|„|‟|″|‴|‶|‷|❝|❞|ʺ|˝'
        self.text = re.sub(regex, "", self.text)

    def remove_apostrophe(self) -> None:
        regex = "‘|’|‚|‛|′|‵|ʹ|ʻ|ʼ|ʽ|ʾ|ʿ|ˈ|ˊ|ˋ"
        self.text = re.sub(regex, "", self.text)

    def remove_hyphen_dash(self) -> None:
        regex = "‑|‒|–|—|―"
        self.text = re.sub(regex, "", self.text)

    def remove_math_symbol(self) -> None:
        regex = "≠|²"
        self.text = re.sub(regex, "", self.text)

    def remove_other_punct(self) -> None:
        self.text = self.text.translate(
            str.maketrans("", "", string.punctuation))

    def remove_space(self) -> None:
        regex = " +"
        self.text = re.sub(regex, " ", self.text).strip()

    def _clean_all(self) -> None:
        self.remove_emoji()
        self.remove_new_line()
        self.remove_number()
        self.remove_mention()
        self.replace_url()
        self.separate_plural()
        self.replace_or()
        self.replace_and()
        self.remove_period_comma()
        self.remove_quotation()
        self.remove_apostrophe()
        self.remove_hyphen_dash()
        self.remove_math_symbol()
        self.remove_other_punct()
        self.remove_space()


def main():
    pass


if __name__ == "__main__":
    main()