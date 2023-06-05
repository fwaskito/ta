# Created Date: Wed, May 31st 2023
# Author: F. Waskito
# Last Modified: Sun, Jun 4th 2023 8:35:03 AM

from typing import Union
from pathlib import Path
import math
import os


def generate_ngram(
    text: str,
    n: int = 1,
) -> list:
    words = text.split(" ")
    if n > 1:
        temp = zip(*[words[i:] for i in range(0, n)])
        ngrams = [" ".join(ngram) for ngram in temp]
        return ngrams
    return words


def get_kamus_path(key: str) -> Union[str, None]:
    root_dir = Path(os.path.dirname(__file__)).parent
    dict_dir = os.path.join(root_dir, "data/dictionary")
    for path in os.scandir(dict_dir):
        if path.is_file() and key in path.name:
            return path.path
    return None


def round_halfup(
    num: float,
    n_digits: int = 0,
) -> Union[float, int]:
    """
    Round a number to a given precision in decimal digits.

    Implementation of 'rounding half up' strategy.
    This strategy to avoid 'rounding bias' when using
    built-in rounding method: round().

    Rounding bias e.g. of 2 decimal points:
        - numbers to round: [2.605, 2.615, 2.625, 2.635, 2.645]
        - expected result : [2.61, 2.62, 2.63, 2.64, 2.65]
        - round() result  : [2.6, 2.62, 2.62, 2.63, 2.65]


    Args:
        num (float): number to round
        n_digits (int): decimal place of

    Returns:
        float: rounded half-up of of float number
    """
    multiplier = 10**n_digits
    return math.floor(num * multiplier + 0.5) / multiplier