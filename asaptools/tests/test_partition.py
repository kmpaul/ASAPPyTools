"""
These are the unit tests for the partition module functions

Copyright 2020, University Corporation for Atmospheric Research
See the LICENSE.txt file for details
"""

import pytest

from asaptools import partition

data = [list(range(3)), list(range(5)), list(range(7))]
index_size = [(0, 1), (1, 3), (5, 9)]
inputs = [(d, i, s) for d in data for i, s in index_size]


def test_out_of_bounds():
    pfunc = partition.EqualLength()
    with pytest.raises(IndexError):
        pfunc([1, 2, 3], index=3, size=3)
    with pytest.raises(IndexError):
        pfunc([1, 2, 3], index=7, size=3)


@pytest.mark.parametrize('data,index,size', inputs)
def test_duplicate(data, index, size):
    pfunc = partition.Duplicate()
    actual = pfunc(data, index=index, size=size)
    assert actual == data


equal_length_results = [
    list(range(3)),
    [1],
    [],
    list(range(5)),
    [2, 3],
    [],
    list(range(7)),
    [3, 4],
    [5],
]
equal_length_inputs = list(i + (r,) for i, r in zip(inputs, equal_length_results))


@pytest.mark.parametrize('data,index,size,results', equal_length_inputs)
def test_equal_length(data, index, size, results):
    pfunc = partition.EqualLength()
    actual = pfunc(data, index=index, size=size)
    assert actual == results


@pytest.mark.parametrize('data,index,size', inputs)
def test_equal_stride(data, index, size):
    pfunc = partition.EqualStride()
    actual = pfunc(data, index=index, size=size)
    expected = list(data[index::size])
    assert actual == expected


@pytest.mark.parametrize('data,index,size', inputs)
def test_sorted_stride(data, index, size):
    weights = [(20 - i) for i in data]
    pfunc = partition.SortedStride()
    actual = pfunc(list(zip(data, weights)), index=index, size=size)
    expected = list(data[:])
    expected.reverse()
    expected = expected[index::size]
    assert actual == expected


weight_balanced_results = [
    set([0, 1, 2]),
    set([1]),
    set(),
    set([3, 2, 4, 1, 0]),
    set([1]),
    set(),
    set([3, 2, 4, 1, 5, 0, 6]),
    set([3, 6]),
    set([4]),
]
weight_balanced_inputs = list(i + (r,) for i, r in zip(inputs, weight_balanced_results))


@pytest.mark.parametrize('data,index,size,results', weight_balanced_inputs)
def test_weight_balanced(data, index, size, results):
    weights = [(3 - i) ** 2 for i in data]
    pfunc = partition.WeightBalanced()
    actual = set(pfunc(list(zip(data, weights)), index=index, size=size))
    assert actual == results
