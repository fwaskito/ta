from pandas import RangeIndex
from pandas import isna, read_csv

class KamusSlang:
    def __init__(self, file_path=None) -> None:
        self._file_path = file_path
        self._kamus = None

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, file_path) -> None:
        self.file_path = file_path

    def get_index(self) -> RangeIndex:
        return self._kamus.index

    def find(self, word) -> bool:
        index = self.get_index()
        if word in index:
            return True
        return False

    def get_meaning(self, word) -> str:
        if self.find(word):
            meaning = self._kamus.loc[word, "Makna"]
            if not isna(meaning):
                return meaning
        return word

    def generate(self) -> None:
        self._kamus = read_csv(
            self._file_path,
            index_col="Slang",
        )