from abc import ABC, abstractmethod
from deep_translator import GoogleTranslator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm import tqdm
# from flair.data import Sentence
# from flair.nn import Classifier


class Labeler(ABC):
    def __init__(self, texts) -> None:
        self.texts = texts
        self._labels = []
        self._polarities = []
        self._translator = GoogleTranslator(
            source='auto',
            target='en',
        )

    @property
    def labels(self) -> list:
        return self._labels

    @property
    def polarities(self) -> list:
        return self._polarities

    @abstractmethod
    def generate(self) -> None:
        pass


class BlobLabeler(Labeler):
    def __init__(self, texts) -> None:
        super().__init__(texts)
        self._subjectivities = []

    @property
    def subjectivities(self) -> list:
        return self._subjectivities

    def _get_sentiment(self, text) -> float:
        blob = TextBlob(text)
        sentiment = blob.sentiment
        return sentiment

    def _convert_polarity(self, polarity) -> str:
        if polarity >= -1 and polarity < 0:
            return 'negative'
        if polarity > 0 and polarity <= 1:
            return 'positive'
        return 'neutral'

    def generate(self) -> None:
        for text in tqdm(self.texts, desc="labeling"):
            text = self._translator.translate(text)
            sentiment = self._get_sentiment(text)
            polarity = sentiment.polarity
            subjectivity = sentiment.subjectivity
            self._polarities.append(polarity)
            self._subjectivities.append(subjectivity)
            self._labels.append(self._convert_polarity(polarity))


class VaderLabeler(Labeler):
    def __init__(self, texts) -> None:
        super().__init__(texts)
        self._vader_sia = SentimentIntensityAnalyzer()

    def _get_compound_polarity(self, text) -> float:
        polarity = self._vader_sia.polarity_scores(text)
        return polarity['compound']

    def _convert_compound_polarity(self, polarity) -> str:
        if polarity <= -0.05:
            return 'negative'
        if polarity >= 0.05:
            return 'positive'
        return 'neutral'

    def generate(self) -> None:
        for text in tqdm(self.texts, desc="labeling"):
            text = self._translator.translate(text)
            polarity = self._get_compound_polarity(text)
            label = self._convert_compound_polarity(polarity)
            self._polarities.append(polarity)
            self._labels.append(label)

# class FlairLabeler(Labeler):
#     def __init__(self, texts) -> None:
#         super().__init__(texts)
#         self._scores = []
#         self._tagger = Classifier.load('sentiment')

#     @property
#     def scores(self) -> list:
#         return self._scores

#     def _get_sentiment_score(self, text) -> list:
#         sentence = Sentence(text)
#         self._tagger.predict(sentence)
#         return sentence.labels

#     def generate(self) -> None:
#         for text in tqdm(self.texts, desc="labeling"):
#             translated_text = self._translator.translate(text)
#             sentiment_score = self._get_sentiment_score(translated_text)
#             score = sentiment_score[0].score
#             label = sentiment_score[0].value.lower()
#             self._scores.append(score)
#             self._labels.append(label)