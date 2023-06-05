# Created Date: Mon, Mar 20th 2023
# Author: F. Waskito
# Last Modified: Sun, Jun 4th 2023 8:41:11 AM

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from numpy import ndarray


class TextVectorizer:
    def __init__(self, corpus):
        self._corpus = corpus
        self._vectors = None
        self._vocabs = None

    @property
    def vectors(self) -> ndarray:
        return self._vectors

    @property
    def vocabs(self) -> tuple:
        return self._vocabs

    def _bow_vectorize(self, min_df, norm: bool) -> None:
        vectorizer = CountVectorizer(min_df=min_df)
        vectors = vectorizer.fit_transform(self._corpus)
        vectors = vectors.toarray()
        if norm:
            scaler = MinMaxScaler()
            self._vectors = scaler.fit_transform(vectors)
        else:
            self._vectors = vectors

        vocabs = tuple(vectorizer.get_feature_names_out())
        self._vocabs = vocabs

    def _tfidf_vectorize(self, min_df, norm) -> None:
        norm_type = None
        if norm:
            norm_type = "l2"

        vectorizer = TfidfVectorizer(
            min_df=min_df,
            norm=norm_type,
            smooth_idf=True,
        )
        vectors = vectorizer.fit_transform(self._corpus)
        self._vectors = vectors.toarray()
        vocabs = tuple(vectorizer.get_feature_names_out())
        self._vocabs = vocabs

    def transform(
        self,
        target: str,
        min_df: int = 1,
        norm: bool = False,
    ) -> None:
        if target == "bow":
            self._bow_vectorize(min_df, norm)
        elif target == "tfidf":
            self._tfidf_vectorize(min_df, norm)