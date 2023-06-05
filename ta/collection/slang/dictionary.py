# Created Date: Sun, Feb 5th 2023
# Author: F. Waskito
# Last Modified: Sun, Jun 4th 2023 8:30:05 AM

from typing import Optional
from pandas import RangeIndex, isna, read_csv
from collection.helper import get_kamus_path


class KamusSlang:
    def __init__(
        self,
        file_path: Optional[str] = None,
    ) -> None:
        self._file_path = file_path
        self._kamus = None

    @property
    def file_path(self) -> str:
        return self._file_path

    def set_file_path(self, path) -> None:
        self.file_path = path

    def get_index(self) -> RangeIndex:
        return self._kamus.index

    def find(self, word) -> bool:
        index = self.get_index()
        if word in index:
            return True
        return False

    def get_meaning(self, word: str) -> str:
        if self.find(word):
            meaning = self._kamus.loc[word, "Makna"]
            if not isna(meaning):
                return meaning
        return word

    def generate(self) -> None:
        if self.file_path is None:
            self._file_path = get_kamus_path("slang")

        self._kamus = read_csv(
            self._file_path,
            index_col="Slang",
        )