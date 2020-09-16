"""
Unit tests (serial only) for the TimeKeeper class

Copyright 2020, University Corporation for Atmospheric Research
See the LICENSE.txt file for details
"""

from time import sleep

from numpy.testing import assert_almost_equal

from asaptools import timekeeper


def test_init():
    tk = timekeeper.TimeKeeper()
    assert isinstance(tk, timekeeper.TimeKeeper)


def test_start_stop_names():
    tk = timekeeper.TimeKeeper()
    name = 'Test Clock'
    wait_time = 0.05
    tk.start(name)
    sleep(wait_time)
    tk.stop(name)
    assert name in tk._accumulated_times
    assert name in tk._added_order
    assert name in tk._start_times


def test_start_stop_values():
    tk = timekeeper.TimeKeeper()
    name = 'Test Clock'
    wait_time = 0.05
    tk.start(name)
    sleep(wait_time)
    tk.stop(name)
    dt = tk.get_time(name)
    dterr = abs(dt / wait_time - 1.0)
    assert dterr < 0.15


def test_start_stop_order_names():
    tk = timekeeper.TimeKeeper()
    name1 = 'Test Clock 1'
    name2 = 'Test Clock 2'
    wait_time = 0.01
    tk.start(name1)
    sleep(wait_time)
    tk.stop(name1)
    tk.start(name2)
    sleep(wait_time)
    tk.stop(name2)
    assert name1 == tk._added_order[0]
    assert name2 == tk._added_order[1]


def test_start_stop_values2():
    tk = timekeeper.TimeKeeper()
    name1 = 'Test Clock 1'
    name2 = 'Test Clock 2'
    wait_time = 0.05
    tk.start(name1)
    sleep(2 * wait_time)
    tk.start(name2)
    sleep(wait_time)
    tk.stop(name1)
    sleep(wait_time)
    tk.stop(name2)
    dt1 = tk.get_time(name1)
    dt1err = abs(dt1 / (3 * wait_time) - 1.0)
    assert dt1err < 0.15
    dt2 = tk.get_time(name2)
    dt2err = abs(dt2 / (2 * wait_time) - 1.0)
    assert dt2err < 0.15


def test_reset_values():
    tk = timekeeper.TimeKeeper()
    name = 'Test Clock'
    wait_time = 0.05
    tk.start(name)
    sleep(wait_time)
    tk.stop(name)
    tk.reset(name)
    assert 0 == tk.get_time(name)


def test_get_time():
    tk = timekeeper.TimeKeeper()
    name = 'Test Clock'
    wait_time = 0.05
    tk.start(name)
    sleep(wait_time)
    tk.stop(name)
    dt = tk.get_time(name)
    dterr = abs(dt / wait_time - 1.0)
    assert dterr < 0.15


def test_get_all_times():
    tk = timekeeper.TimeKeeper()
    name1 = 'Test Clock 1'
    name2 = 'Test Clock 2'
    wait_time = 0.05
    tk.start(name1)
    sleep(2 * wait_time)
    tk.start(name2)
    sleep(wait_time)
    tk.stop(name1)
    sleep(wait_time)
    tk.stop(name2)
    all_times = tk.get_all_times()
    expected_all_times = {name1: 3 * wait_time, name2: 2 * wait_time}
    assert list(expected_all_times.keys()) == list(all_times.keys())
    assert_almost_equal(list(expected_all_times.values()), list(all_times.values()), decimal=2)
