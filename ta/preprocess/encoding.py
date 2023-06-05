# Created Date: Sun, May 28th 2023
# Author: F. Waskito
# Last Modified: Sun, Jun 4th 2023 8:43:17 AM

from numpy import ndarray, array


class LabelEncoder:
    def __init__(self, labels: list) -> None:
        self._labels = labels
        self._classes = sorted(list(set(labels)))
        self._encoded_labels = []

    @property
    def encoded_labels(self) -> ndarray:
        return array(self._encoded_labels)

    def _encode_integer(self) -> None:
        for label in self._labels:
            label = self._classes.index(label)
            self._encoded_labels.append(label)

    def _encode_binary(self) -> None:
        for label in self._labels:
            index = self._classes.index(label)
            label = [0] * len(self._classes)
            label[index] = 1
            self._encoded_labels.append(label)

    def transform(self, target: str = "integer") -> None:
        self._encoded_labels = []
        if target == "integer":
            self._encode_integer()
        elif target == "binary":
            self._encode_binary()