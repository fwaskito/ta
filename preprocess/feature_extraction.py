from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class TextVectorizer:
    def __init__(self, corpus):
        self._corpus = corpus
        self._vectors = None
        self._vocabs = None

    @property
    def vectors(self):
        return self._vectors

    @property
    def vocabs(self):
        return self._vocabs

    def _bow_vectorize(self, min_df, norm):
        vectorizer = CountVectorizer(min_df=min_df)
        vectors = vectorizer.fit_transform(self._corpus)
        vectors = vectors.toarray()
        if norm:
            max_val = vectors.max()
            min_val = vectors.min()
            self._vectors = (vectors) / (max_val - min_val)
        else:
            self._vectors = vectors

        vocabs = tuple(vectorizer.get_feature_names_out())
        self._vocabs = vocabs

    def _tfidf_vectorize(self, min_df, norm, smooth_idf=True):
        norm_type = None
        if norm:
            norm_type = "l2"

        vectorizer = TfidfVectorizer(
            min_df=min_df,
            norm=norm_type,
            smooth_idf=smooth_idf,
        )
        vectors = vectorizer.fit_transform(self._corpus)
        self._vectors = vectors.toarray()
        vocabs = tuple(vectorizer.get_feature_names_out())
        self._vocabs = vocabs

    def transform(self, target="bow", min_df=1, norm=False):
        if target == "bow":
            self._bow_vectorize(min_df, norm)
        elif target == "tfidf":
            self._tfidf_vectorize(min_df, norm)
