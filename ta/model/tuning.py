# Created Date: Sun, Jun 4th 2023
# Author: F. Waskito
# Last Modified: Sun, Jun 4th 2023 7:07:03 AM

from typing import Optional
from itertools import product
from numpy import ndarray
from tqdm import tqdm
from sklearn.svm import SVC
from validation.cross import ImbalancedCV


class SVMGridSearchCV:
    def __init__(
        self,
        model: SVC,
        params: dict,
        cv: int,
        scoring: list[str],
        scoring_avg: str = "weighted",
    ) -> None:
        self._model = model
        self._params = params
        self._cv = cv
        self._scoring = scoring
        self._scoring_avg = scoring_avg
        self.random_state: Optional[int] = None
        self.verbose: int = 2
        self._evaluation_result = []
        self._kernel = None
        self._param_keys = []
        self._param_combinations = None
        self._X = None
        self._y = None

    @property
    def evaluation_result(self) -> tuple:
        return tuple(self._evaluation_result)

    def get_sequential_result(self) -> dict[list]:
        sequential_result = {}
        for key in self._param_keys:
            sequential_result[key] = []

        for key in self._scoring:
            sequential_result[f"mean_{key}"] = []

        for result in self._evaluation_result:
            for key, value in result["params"].items():
                sequential_result[key].append(value)

            for key, value in result["scores"].items():
                sequential_result[key].append(value)
        return sequential_result

    def get_best_result(
        self,
        base_on: str = "f1",
        n: int = 1,
    ) -> dict:
        base_on = f"mean_{base_on}"
        return sorted(
            self._evaluation_result,
            key=lambda x: x['scores'][base_on],
            reverse=True,
        )[:n]

    def get_worst_result(
        self,
        base_on: str = "f1",
        n: int = 1,
    ) -> dict:
        base_on = f"mean_{base_on}"
        return list(
            reversed(
                sorted(
                    self._evaluation_result,
                    key=lambda x: x['scores'][base_on],
                    reverse=True,
                )[-(n):]))

    def _validate(self, set_params) -> dict:
        validator = ImbalancedCV(
            model=self._model.set_params(**set_params),
            n_fold=self._cv,
            scoring=self._scoring,
            scoring_avg=self._scoring_avg,
            random_state=self.random_state,
        )
        validator.progress_bar = False
        validator.validate(self._X, self._y)
        score = validator.get_score()
        return score

    def _show_nparam_combination(self) -> None:
        if self.verbose > 0:
            n = len(self._param_combinations)
            print(f"Total combination of parameters: {n}")

    def _show_new_result(self, i, set_params, scores) -> None:
        if self.verbose > 1:
            print(f"i={i}|{set_params}  {scores}")

    def _search(self) -> None:
        self._show_nparam_combination()
        progress_bar = self.verbose > 2
        for i, set_params in enumerate(
                tqdm(
                    self._param_combinations,
                    desc="search",
                    disable=not progress_bar,
                )):
            scores = self._validate(set_params)
            self._evaluation_result.append({
                "params": set_params,
                "scores": scores,
            })
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

    def fit(self, X: ndarray, y: ndarray) -> None:
        self._X = X
        self._y = y
        self._kernel = self._model.get_params()["kernel"]
        self._param_combinations = self._combine_params()
        self._search()