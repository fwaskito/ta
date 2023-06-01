from typing import Optional
from statistics import mean
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from numpy import ndarray
from tqdm import tqdm


class ImbalanceCV:
    def __init__(
        self,
        model: object,
        n_splits: int = 5,
        random_state: Optional[int] = None,
        score_average: str = "weighted",
    ) -> None:
        self._model = model
        self._n_splits = n_splits
        self._score_average = score_average
        self.random_state = random_state
        self.progress_bar: bool = True
        self._smoter = SMOTE(random_state=random_state)
        self._result = {
            "acc": [],
            "pre": [],
            "rec": [],
            "f1": [],
        }
        self._X = None
        self._y = None

    def get_score(self, full=False) -> dict:
        if full:
            return {
                "accuracy": self._result["acc"],
                "precison": self._result["pre"],
                "recall": self._result["rec"],
                "f1": self._result["f1"],
            }
        mean_acc = mean(self._result["acc"])
        mean_pre = mean(self._result["pre"])
        mean_rec = mean(self._result["rec"])
        mean_f1 = mean(self._result["f1"])
        return {
            "mean_accuracy": round(mean_acc, 3),
            "mean_precison": round(mean_pre, 3),
            "mean_recall": round(mean_rec, 3),
            "mean_f1": round(mean_f1, 3),
        }

    def _score_fold(self, y, y_pred) -> None:
        if self._score_average == "weighted":
            pre = precision_score(y, y_pred, average="weighted")
            rec = recall_score(y, y_pred, average="weighted")
            f1_ = f1_score(y, y_pred, average="weighted")
        elif self._score_average == "macro":
            pre = precision_score(y, y_pred, average="macro")
            rec = recall_score(y, y_pred, average="macro")
            f1_ = f1_score(y, y_pred, average="macro")

        acc = accuracy_score(y, y_pred)
        self._result["acc"].append(acc)
        self._result["pre"].append(pre)
        self._result["rec"].append(rec)
        self._result["f1"].append(f1_)

    def _train_val_split(
        self,
        train_index,
        val_index,
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        train_index = list(train_index)
        val_index = list(val_index)
        # get training and validation set
        X_train = self._X[train_index]
        y_train = self._y[train_index]
        X_val = self._X[val_index]
        y_val = self._y[val_index]
        return X_train, X_val, y_train, y_val

    def validate(self, X: ndarray, y: ndarray) -> None:
        self._X = X
        self._y = y
        k_fold = KFold(
            n_splits=self._n_splits,
            random_state=self.random_state,
            shuffle=True,
        )
        # K-Fold cross validation stage
        for train_fold_index, val_fold_index in tqdm(
                k_fold.split(X, y),
                total=self._n_splits,
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
            # scoring
            model = self._model
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            self._score_fold(y_val, y_val_pred)