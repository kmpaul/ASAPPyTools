"""
Tests of the verbose printer utility

Copyright 2020, University Corporation for Atmospheric Research
See the LICENSE.txt file for details
"""

from os import linesep

import pytest

from asaptools.vprinter import VPrinter


@pytest.fixture
def header():
    return '[1] '


@pytest.fixture
def vprinter(header):
    return VPrinter(header=header, verbosity=2)


@pytest.mark.parametrize('use_header', [False, True])
def test_to_str(header, vprinter, use_header):
    data = ['a', 'b', 'c', 1, 2, 3, 4.0, 5.0, 6.0]
    actual = vprinter.to_str(*data, header=use_header)
    expected = header if use_header else ''
    expected += ''.join([str(d) for d in data])
    assert actual == expected


@pytest.mark.parametrize('use_header', [False, True])
@pytest.mark.parametrize('verbosity', [1, 3])
def test_vprinter(capsys, vprinter, use_header, verbosity):
    data = ['a', 'b', 'c', 1, 2, 3, 4.0, 5.0, 6.0]
    vprinter(*data, header=use_header, verbosity=verbosity)
    actual = capsys.readouterr().out
    if verbosity > 2:
        expected = ''
    else:
        expected = vprinter.to_str(*data, header=use_header) + linesep
    assert actual == expected
