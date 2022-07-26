import pytest
from numpy import array_equal

from serenityff.charge.gnn.utils.split_utils import get_split_numbers, split_data_Kfold, split_data_random


@pytest.fixture
def data():
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_get_split_numbers():
    assert [50, 50] == get_split_numbers(N=100, train_ratio=0.5)
    assert [51, 50] == get_split_numbers(N=101, train_ratio=0.5)
    assert [51, 51] == get_split_numbers(N=102, train_ratio=0.5)


def test_random_split(data):
    train, test = split_data_random(data_list=data, train_ratio=0.5)
    assert len(train) == 6
    assert len(test) == 5


def test_kfold_split(data):
    train1, test1 = split_data_Kfold(data, n_splits=2, split=0)
    train2, test2 = split_data_Kfold(data, n_splits=2, split=1)
    assert not array_equal(train1, train2)
    assert not array_equal(test1, test2)
