# Created Date: Sun, Feb 5th 2023
# Author: F. Waskito
# Last Modified: Sun, Jun 4th 2023 8:35:26 AM

from re import search
from time import sleep
from pandas import DataFrame
from snscrape.modules.twitter import TwitterSearchScraper
from googletrans import Translator
from tqdm import tqdm


class TweetScraper:
    def __init__(
        self,
        keyword: str,
        lang: str,
        since: str,
        until: str,
    ) -> None:
        self.keyword = keyword
        self.lang = lang
        self.since = since
        self.until = until
        self._tweet_attr = [
            "Tweet_ID",
            "Datetime",
            "Username",
            "Text",
        ]
        self._tweets_table = None
        self._n_tweets = 0

    @property
    def n_tweets(self) -> int:
        return self._n_tweets

    @property
    def tweets_table(self) -> DataFrame:
        return self._tweets_table

    def scrape(self) -> None:
        query = f"{self.keyword} lang:{self.lang}"
        query += f" since:{self.since} until:{self.until}"
        scraper = TwitterSearchScraper(query)
        self._n_tweets = len(list(scraper.get_items()))
        tweets_table = DataFrame(columns=self._tweet_attr)
        for i, tweet in enumerate(
                tqdm(
                    scraper.get_items(),
                    total=self._n_tweets,
                    desc="Scraping",
                )):
            tweets_table.loc[i] = [
                tweet.id,
                tweet.date,
                tweet.user.username,
                tweet.rawContent,
            ]
            sleep(0.001)

        tweets_table[::-1].reset_index(drop=True, inplace=True)
        self._tweets_table = tweets_table

    def _find_keyword(self, text) -> bool:
        regex = self.keyword.replace(" AND ", "|")
        regex = regex.replace(" OR ", "|")
        if search(regex, text.lower()):
            return True
        return False

    def _detect_lang(self, text) -> str:
        translator = Translator()
        return translator.detect(text).lang

    def remove_irrelevant(self) -> DataFrame:
        columns = self._tweet_attr
        relevant_tweets_table = DataFrame(columns=columns)
        irrelevant_tweets_table = DataFrame(columns=columns)
        for i, text in enumerate(
                tqdm(
                    self._tweets_table["Text"],
                    total=self._n_tweets,
                    desc="Removing irrelevant",
                )):
            tweet = self._tweets_table.loc[i]
            if self._find_keyword(text) and self._detect_lang(
                    text) == self.lang:
                index = len(relevant_tweets_table)
                relevant_tweets_table.loc[index] = tweet
            else:
                index = len(irrelevant_tweets_table)
                irrelevant_tweets_table.loc[index] = tweet

            sleep(0.001)
        self._n_tweets = len(relevant_tweets_table)
        self._tweets_table = relevant_tweets_table
        return irrelevant_tweets_table