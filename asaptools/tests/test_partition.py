"""
These are the unit tests for the partition module functions

Copyright 2020, University Corporation for Atmospheric Research
See the LICENSE.txt file for details
"""

import pytest
from numpy import array, stack, testing

from asaptools import partition

ditems = [range(3), range(5), range(7)]
index_size = [(0, 1), (1, 3), (5, 9)]
inputs = [(d, i, s) for d in ditems for i, s in index_size]


@pytest.mark.parametrize(
    'pfunc,index',
    [
        (partition.EqualLength(), 3),
        (partition.EqualStride(), 7),
    ],
)
@pytest.mark.parametrize('dtype', [list, array])
def test_out_of_bounds(pfunc, index, dtype):
    data = dtype([1, 2, 3])
    with pytest.raises(IndexError):
        pfunc(data, index=index, size=3)


@pytest.mark.parametrize('dtype', [list, array])
@pytest.mark.parametrize('ditem,index,size', inputs)
def test_duplicate(ditem, index, size, dtype):
    data = dtype(ditem)
    pfunc = partition.Duplicate()
    actual = pfunc(data, index=index, size=size)
    testing.assert_equal(actual, data)


equal_length_results = [
    (list(range(3)),),
    ([1],),
    ([],),
    (list(range(5)),),
    ([2, 3],),
    ([],),
    (list(range(7)),),
    ([3, 4],),
    ([5],),
]
equal_length_inputs = list(i + r for i, r in zip(inputs, equal_length_results))


@pytest.mark.parametrize('dtype', [list, array])
@pytest.mark.parametrize('ditem,index,size,results', equal_length_inputs)
def test_equal_length(dtype, ditem, index, size, results):
    data = dtype(ditem)
    pfunc = partition.EqualLength()
    actual = pfunc(data, index=index, size=size)
    expected = dtype(results)
    testing.assert_equal(actual, expected)


@pytest.mark.parametrize('dtype', [list, array])
@pytest.mark.parametrize('ditem,index,size', inputs)
def test_equal_stride(dtype, ditem, index, size):
    data = dtype(ditem)
    pfunc = partition.EqualStride()
    actual = pfunc(data, index=index, size=size)
    expected = data[index::size]
    testing.assert_equal(actual, expected)


@pytest.mark.parametrize('dtype', [list, array])
@pytest.mark.parametrize('ditem,index,size', inputs)
def test_sorted_stride(dtype, ditem, index, size):
    data = dtype(ditem)
    weights = [(20 - i) for i in data]
    pfunc = partition.SortedStride()
    actual = pfunc(list(zip(data, weights)), index=index, size=size)
    expected = list(ditem[:])
    expected.reverse()
    expected = dtype(expected[index::size])
    testing.assert_equal(actual, expected)


weight_balanced_results = [
    [0, 1, 2],
    [1],
    [],
    [0, 1, 2, 4, 3],
    [1],
    [],
    [0, 6, 1, 5, 2, 4, 3],
    [6, 3],
    [4],
]
weight_balanced_inputs = list(i + (r,) for i, r in zip(inputs, weight_balanced_results))


@pytest.mark.parametrize('ditem,index,size,results', weight_balanced_inputs)
def test_weight_balanced_lists(ditem, index, size, results):
    data = list(ditem)
    weights = [(3 - i) ** 2 for i in data]
    pfunc = partition.WeightBalanced()
    actual = pfunc(list(zip(data, weights)), index=index, size=size)
    testing.assert_equal(actual, results)


@pytest.mark.parametrize('ditem,index,size,results', weight_balanced_inputs)
def test_weight_balanced_arrays(ditem, index, size, results):
    data = array(ditem)
    weights = array([(3 - i) ** 2 for i in data])
    data_wgts = stack((data, weights)).transpose()
    pfunc = partition.WeightBalanced()
    actual = pfunc(data_wgts, index=index, size=size)
    testing.assert_equal(actual, array(results))
