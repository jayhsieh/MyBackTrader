#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2015-2020 Daniel Rodriguez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
import itertools
import sys


try:
    import winreg
except ImportError:
    winreg = None

MAXINT = sys.maxsize
MININT = -sys.maxsize - 1

MAXFLOAT = sys.float_info.max
MINFLOAT = sys.float_info.min

string_types = str,
integer_types = int,

filter = filter
map = map
range = range
zip = zip
long = int

def cmp(a, b): return (a > b) - (a < b)

def bytes(x): return x.encode('utf-8')

def bstr(x): return str(x)

from io import StringIO

from urllib.request import (urlopen, ProxyHandler, build_opener,
                            install_opener)
from urllib.parse import quote as urlquote

def iterkeys(d): return iter(d.keys())

def itervalues(d): return iter(d.values())

def iteritems(d): return iter(d.items())

def keys(d): return list(d.keys())

def values(d): return list(d.values())

def items(d): return list(d.items())

import queue as queue


# This is from Armin Ronacher from Flash simplified later by six
# 偽元類: 不是元類，但是實現元類的功能
def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):
        # __new__ 函數中運行代碼 MetaParams(name, bases, dict) 來生成 cerebro
        # 而 MetaParams 繼承父類 MetaBase
        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    # 輸出以 metaclass 為類型的類，暫時命名為 return_class
    return type.__new__(metaclass, str('temporary_class'), (), {})
