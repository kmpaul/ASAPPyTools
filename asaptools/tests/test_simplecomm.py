"""
Tests for the SimpleComm class

Copyright 2020, University Corporation for Atmospheric Research
See the LICENSE.txt file for details
"""

import subprocess
import sys

import pytest
from jinja2 import Environment, FileSystemLoader

pytest.importorskip('mpi4py')

env = Environment(loader=FileSystemLoader('asaptools/tests/templates'))


@pytest.mark.parametrize('example', ['basic', 'sum', 'max'])
@pytest.mark.parametrize('n', [0, 1, 4])
@pytest.mark.parametrize('serial', [True, False])
def test_simplecomm_allreduce(tmpdir, example, n, serial):
    template = env.get_template(example + '.py.j2')
    script_text = template.render(serial=serial)
    script_file = tmpdir.join(example + '.py')
    script_file.write(script_text)

    cmds = [] if n == 0 else ['mpirun', '-n', str(n)]
    p = subprocess.Popen(cmds + [sys.executable, script_file])

    p.communicate()
    assert p.returncode == 0


@pytest.mark.parametrize('example', ['partition'])
@pytest.mark.parametrize('n', [1, 4])
@pytest.mark.parametrize('involved', [True, False])
def test_simplecomm_partition(tmpdir, example, n, involved):
    template = env.get_template(example + '.py.j2')
    script_text = template.render(involved=involved)
    script_file = tmpdir.join(example + '.py')
    script_file.write(script_text)

    p = subprocess.Popen(['mpirun', '-n', str(n), sys.executable, script_file])

    p.communicate()
    assert p.returncode == 0
