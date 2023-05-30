import re
import time
from tqdm import tqdm
from pandas import DataFrame
from snscrape.modules.twitter import TwitterSearchScraper
from googletrans import Translator


class TweetScraper:
    def __init__(self, keyword, lang, since, until):
        self.keyword = keyword
        self.lang = lang
        self.since = since
        self.until = until
        self._tweet_attr = ["Tweet_ID", "Datetime", "Username", "Text"]
        self._tweets_table = DataFrame(columns=self._tweet_attr)
        self._num_of_tweets = 0

    @property
    def num_of_tweets(self):
        return self._num_of_tweets

    @property
    def tweets_table(self):
        return self._tweets_table

    def scrape(self):
        query = self.keyword + " lang:" + self.lang
        query += " since:" + self.since + " until:" + self.until
        scraper = TwitterSearchScraper(query)
        self._num_of_tweets = len(list(scraper.get_items()))
        for i, tweet in enumerate(
                tqdm(
                    scraper.get_items(),
                    total=self._num_of_tweets,
                    desc="Scraping",
                )):
            self._tweets_table.loc[i] = [
                tweet.id,
                tweet.date,
                tweet.user.username,
                tweet.rawContent,
            ]
            time.sleep(0.001)

        # reverse tweets order
        self._tweets_table = self._tweets_table[::-1].reset_index(drop=True)

    def _find_keyword(self, text) -> bool:
        regex = self.keyword.replace(" AND ", "|").replace(" OR ", "|")
        if re.search(regex, text.lower()):
            return True
        return False

    def _detect_language(self, text) -> str:
        translator = Translator()
        return translator.detect(text).lang

    def remove_irrelevant(self):
        relevant_tweets_table = DataFrame(columns=self._tweet_attr)
        irrelevant_tweets_table = DataFrame(columns=self._tweet_attr)
        for i, text in enumerate(
                tqdm(
                    self._tweets_table["Text"],
                    total=self._num_of_tweets,
                    desc="Removing irrelevant",
                )):
            tweet = self._tweets_table.loc[i]
            if self._find_keyword(text) and self._detect_language(
                    text) == self.lang:
                new_idx = len(relevant_tweets_table)
                relevant_tweets_table.loc[new_idx] = tweet
            else:
                new_idx = len(irrelevant_tweets_table)
                irrelevant_tweets_table.loc[new_idx] = tweet
            time.sleep(0.001)

        self._num_of_tweets = len(relevant_tweets_table)
        self._tweets_table = relevant_tweets_table.copy()
        return irrelevant_tweets_table
