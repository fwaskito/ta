# Created Date: Mon, May 29th 2023
# Author: F. Waskito
# Last Modified: Sun, Jun 4th 2023 8:25:11 AM

from typing import Optional, Union
from pandas import RangeIndex, isna, read_csv
from collection.helper import get_kamus_path


class KamusAntonim:
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

    def get_antonym(self, word) -> Union[str, None]:
        if self.find(word):
            antonym = self._kamus.loc[word, "Antonim"]
            if not isna(antonym):
                return antonym
        return None

    def generate(self):
        if self.file_path is None:
            self._file_path = get_kamus_path("antonim")

        self._kamus = read_csv(
            self._file_path,
            index_col="Kata",
        )