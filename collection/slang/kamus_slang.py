import pandas as pd


class KamusSlang:
    def __init__(self, file_path=None) -> None:
        self._file_path = file_path
        self._kamus = None

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, file_path):
        self.file_path = file_path

    def get_index(self):
        return self._kamus.index

    def find(self, word):
        index = self.get_index()
        if word in index:
            return True
        return False

    def get_meaning(self, word):
        if self.find(word):
            meaning = self._kamus.loc[word, "Makna"]
            if not pd.isna(meaning):
                return meaning
        return word

    def generate(self):
        self._kamus = pd.read_csv(
            self._file_path,
            index_col="Slang",
        )