# Created Date: Fri, May 19th 2023
# Author: F. Waskito
# Last Modified: Sat, Jun 3rd 2023 9:18:19 AM

from typing import Optional, Tuple
from statistics import mean
from numpy import ndarray
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score
from collection.helper import round_halfup


class ImbalancedCV:
    def __init__(
        self,
        model: object,
        n_fold: int,
        scoring: list[str],
        scoring_avg: str = "weighted",
        random_state: Optional[int] = None,
    ) -> None:
        self._model = model
        self._n_fold = n_fold
        self._scoring = scoring
        self._scoring_avg = scoring_avg
        self.random_state = random_state
        self.progress_bar: bool = True
        self._smoter = SMOTE(random_state=random_state)
        self._result = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }
        self._X = None
        self._y = None

    def get_score(self, full: bool = False) -> dict:
        res_score = {}
        if full:
            for score_type, scores in self._result.items():
                if scores:
                    scores = [round_halfup(num, 3) for num in scores]
                    res_score[score_type] = scores

        for score_type, scores in self._result.items():
            if len(scores) > 0:
                mean_scores = round_halfup(mean(scores), 3)
                res_score[f"mean_{score_type}"] = mean_scores
        return res_score

    def _score_fold(self, y, y_pred) -> None:
        if "accuracy" in self._scoring:
            acc = accuracy_score(y, y_pred)
            self._result["accuracy"].append(acc)

        average = self._scoring_avg
        if "precision" in self._scoring:
            pre = precision_score(y, y_pred, average=average)
            self._result["precision"].append(pre)

        if "recall" in self._scoring:
            rec = recall_score(y, y_pred, average=average)
            self._result["recall"].append(rec)

        if "f1" in self._scoring:
            f1 = f1_score(y, y_pred, average=average)
            self._result["f1"].append(f1)

    def _train_val_split(
        self,
        train_index,
        val_index,
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        train_index = list(train_index)
        val_index = list(val_index)

        # get training and validation set
        X_train = self._X[train_index]
        y_train = self._y[train_index]
        X_val = self._X[val_index]
        y_val = self._y[val_index]
        return X_train, X_val, y_train, y_val

    def validate(
        self,
        X: ndarray,
        y: ndarray,
    ) -> None:
        self._X = X
        self._y = y
        k_fold = KFold(
            n_splits=self._n_fold,
            random_state=self.random_state,
            shuffle=True,
        )
        # K-Fold Cross Calidation stage
        for train_fold_index, val_fold_index in tqdm(
                k_fold.split(X, y),
                total=self._n_fold,
                desc="CV",
                disable=not self.progress_bar,
        ):
            # get instances for current fold
            train_val = self._train_val_split(
                train_fold_index,
                val_fold_index,
            )
            X_train, X_val, y_train, y_val = train_val

            # oversampling
            X_train, y_train = self._smoter.fit_resample(
                X_train,
                y_train,
            )
            # scoring fold
            model = self._model
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            self._score_fold(y_val, y_val_pred)