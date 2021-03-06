#!/usr/bin/env python3
# flake8: noqa
"""
The ASAP Python Toolbox

The ASAP Python Toolbox is a collection of stand-alone tools for doing simple
tasks, from managing print messages with a set verbosity level, to
keeping timing information, to managing simple MPI communication.

Copyright 2020 University Corporation for Atmospheric Research

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Send questions and comments to Kevin Paul (kpaul@ucar.edu).
"""

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # pragma: no cover
    __version__ = '0.0.0'  # pragma: no cover
