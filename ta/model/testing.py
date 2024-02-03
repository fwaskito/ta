# Created Date: Sat, Jan 20th 2024
# Author: F. Waskito
# Last Modified: Tue, Jan 23rd 2024 5:48:33 AM

from typing import Optional
from itertools import product
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import confusion_matrix
from collection.helper import round_halfup


class SVMGridTester:
    def __init__(
        self,
        model: SVC,
        params: dict,
        scoring: list[str],
        scoring_avg: str = "weighted",
        random_state: Optional[int] = None,
    ) -> None:
        self._model = model
        self._params = params
        self._scoring = scoring
        self._scoring_avg = scoring_avg
        self._random_state = random_state
        self.verbose: int = 2
        self._evaluation_result = []
        self._confusion_matrices = []
        self._kernel = None
        self._param_keys = []
        self._param_combinations = None
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

    @property
    def evaluation_result(self) -> tuple:
        return tuple(self._evaluation_result)

    @property
    def confusion_matrices(self) -> tuple:
        return tuple(self._confusion_matrices)

    def get_table_result(self) -> DataFrame:
        sequential_result = {}
        for key in self._param_keys:
            sequential_result[key] = []

        for key in self._scoring:
            sequential_result[key] = []

        for result in self._evaluation_result:
            for key, value in result["params"].items():
                sequential_result[key].append(value)

            for key, value in result["scores"].items():
                sequential_result[key].append(value)
        return DataFrame(sequential_result)

    def get_best_table_result(
        self,
        base_on: str = "f1",
        n: int = 1,
    ) -> DataFrame:
        best_result = self.get_best_result(base_on, n)
        sequential_result = {}
        for key in self._param_keys:
            sequential_result[key] = []

        for key in self._scoring:
            sequential_result[key] = []

        for result in best_result:
            for key, value in result["params"].items():
                sequential_result[key].append(value)

            for key, value in result["scores"].items():
                sequential_result[key].append(value)
        return DataFrame(sequential_result)

    def get_best_result(
        self,
        base_on: str = "f1",
        n: int = 1,
    ) -> dict:
        return sorted(
            self._evaluation_result,
            key=lambda x: x["scores"][base_on],
            reverse=True,
        )[:n]

    def get_worst_result(
        self,
        base_on: str = "f1",
        n: int = 1,
    ) -> dict:
        return list(
            reversed(
                sorted(
                    self._evaluation_result,
                    key=lambda x: x["scores"][base_on],
                    reverse=True,
                )[-(n):]
            )
        )

    def _test(self, set_params) -> dict:
        tester = ImbalancedTester(
            model=self._model.set_params(**set_params),
            scoring=self._scoring,
            scoring_avg=self._scoring_avg,
            random_state=self._random_state,
        )
        tester.test(
            self._X_train,
            self._X_test,
            self._y_train,
            self._y_test,
        )
        score = tester.get_score()
        cm = tester.confusin_matrix
        return score, cm

    def _show_initialization(self) -> None:
        if self.verbose > 0:
            n = len(self._param_combinations)
            print(f"Total combination of parameters: {n}")

    def _show_new_result(self, i, set_params, scores) -> None:
        if self.verbose > 1:
            print(f"i={i}|{set_params}  {scores}")

    def _search(self) -> None:
        self._show_initialization()
        progress_bar = self.verbose > 2
        for i, set_params in enumerate(
            tqdm(
                self._param_combinations,
                desc="predict",
                disable=not progress_bar,
            )
        ):
            scores, cm = self._test(set_params)
            self._evaluation_result.append(
                {
                    "params": set_params,
                    "scores": scores,
                    "cm": cm,
                }
            )
            self._show_new_result(i, set_params, scores)

    def _combine_params(self) -> list[dict]:
        param_values = []
        for key, values in self._params.items():
            self._param_keys.append(key)
            param_values.append(values)

        param_combinations = []
        for combin in product(*param_values):
            set_param = {"kernel": self._kernel}
            for i, param in enumerate(combin):
                set_param[self._param_keys[i]] = param

            param_combinations.append(set_param)
        self._param_keys.insert(0, "kernel")
        return param_combinations

    def fit_predict(
        self,
        X_train: ndarray,
        X_test: ndarray,
        y_train: ndarray,
        y_test: ndarray,
    ) -> None:
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._kernel = self._model.get_params()["kernel"]
        self._param_combinations = self._combine_params()
        self._search()


class ImbalancedTester:
    def __init__(
        self,
        model: object,
        scoring: list[str],
        scoring_avg: str = "weighted",
        random_state: Optional[int] = None,
    ) -> None:
        self._model = model
        self._scoring = scoring
        self._scoring_avg = scoring_avg
        self._random_state = random_state
        self._smoter = SMOTE(random_state=random_state)
        self._result = {}
        self._confusion_matrix = None

    def get_score(self) -> dict:
        res_score = {}
        for score_type, score in self._result.items():
            score = round_halfup(score, 3)
            res_score[score_type] = score
        return res_score

    @property
    def confusin_matrix(self) -> ndarray:
        return self._confusion_matrix

    def _score(self, y_test, y_pred) -> None:
        if "accuracy" in self._scoring:
            acc = accuracy_score(y_test, y_pred)
            self._result["accuracy"] = acc

        avg = self._scoring_avg
        if "precision" in self._scoring:
            pre = precision_score(y_test, y_pred, average=avg)
            self._result["precision"] = pre

        if "recall" in self._scoring:
            rec = recall_score(y_test, y_pred, average=avg)
            self._result["recall"] = rec

        if "f1" in self._scoring:
            f1 = f1_score(y_test, y_pred, average=avg)
            self._result["f1"] = f1

    def _create_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        self._confusion_matrix = cm

    def test(
        self,
        X_train: ndarray,
        X_test: ndarray,
        y_train: ndarray,
        y_test: ndarray,
    ) -> None:
        # oversampling
        X_train, y_train = self._smoter.fit_resample(
            X_train,
            y_train,
        )
        # predicting
        model = self._model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self._score(y_test, y_pred)
        self._create_confusion_matrix(y_test, y_pred)