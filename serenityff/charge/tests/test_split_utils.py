from typing import Sequence

import pytest
from numpy import array_equal

from serenityff.charge.gnn.utils.split_utils import get_split_numbers, split_data_Kfold, split_data_random


@pytest.fixture
def data() -> Sequence[int]:
    return [*range(11)]


def test_get_split_numbers() -> None:
    assert [50, 50] == get_split_numbers(N=100, train_ratio=0.5)
    assert [51, 50] == get_split_numbers(N=101, train_ratio=0.5)
    assert [51, 51] == get_split_numbers(N=102, train_ratio=0.5)
    return


def test_random_split(data) -> None:
    train, test = split_data_random(data_list=data, train_ratio=0.5)
    assert len(train) == 6
    assert len(test) == 5
    return


def test_kfold_split(data) -> None:
    train1, test1 = split_data_Kfold(data, n_splits=2, split=0)
    train2, test2 = split_data_Kfold(data, n_splits=2, split=1)
    assert not array_equal(train1, train2)
    assert not array_equal(test1, test2)
    return
