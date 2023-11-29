# Created Date: Sat, Jun 17th 2023
# Author: F. Waskito
# Last Modified: Tue, Sep 19th 2023 01:48:01 PM

from typing import Union


class TextVectorInspector:
    def __init__(self, vectors, vocabs) -> None:
        self._vectors = vectors
        self._vocabs = vocabs
        self._sum_val_dict = {}
        self._df_dict = {}
        self._count()

    def get_sum_val(self, term: str) -> Union[int, float, None]:
        if term in self._vocabs:
            return self._sum_val_dict[term]
        return None

    def get_df(self, term: str) -> Union[int, None]:
        if term in self._vocabs:
            return self._df_dict[term]
        return None

    def get_min_df(self) -> list:
        return sorted(set(self._df_dict.values()))[0]

    def get_index_term(self, term: str) -> int:
        for i, term_i in enumerate(self._vocabs):
            if term_i == term:
                return i

    def get_nterms_df(self, target_df: int) -> int:
        nterm_df = 0
        for df in self._df_dict.values():
            if target_df == df:
                nterm_df += 1
        return nterm_df

    def get_terms_df(self, target_df: int) -> list:
        term_df = []
        for term, df in self._df_dict.items():
            if target_df == df:
                term_df.append(term)
        return term_df

    def _count(self) -> None:
        for i in range(self._vectors.shape[1]):
            sum_val_i = 0  # if given a BOW vector, equals to 'TF' i
            df_i = 0  # document frequency of term i
            for j in range(self._vectors.shape[0]):
                val_ij = self._vectors[j][i]
                sum_val_i += val_ij
                if val_ij != 0:
                    df_i += 1

            self._df_dict[self._vocabs[i]] = df_i
            self._sum_val_dict[self._vocabs[i]] = sum_val_i