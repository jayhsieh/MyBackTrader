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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import math
import pytz
import time as _time

from .py3 import string_types

TZ_AU = pytz.timezone('Australia/Sydney')
"""Time zone of Australia market"""
TZ_TP = pytz.timezone('Asia/Taipei')
"""Time zone of Taipei market"""
TZ_US = pytz.timezone('America/New_York')
"""Time zone of US market"""
TZ_UTC = pytz.timezone('UTC')

ZERO = datetime.timedelta(0)

STDOFFSET = datetime.timedelta(seconds=-_time.timezone)
if _time.daylight:
    DSTOFFSET = datetime.timedelta(seconds=-_time.altzone)
else:
    DSTOFFSET = STDOFFSET

DSTDIFF = DSTOFFSET - STDOFFSET

# To avoid rounding errors taking dates to next day
TIME_MAX = datetime.time(23, 59, 59, 999990)

# To avoid rounding errors taking dates to next day
TIME_MIN = datetime.time.min

DT_MIN = datetime.datetime.min


def tzparse(tz):
    # If no object has been provided by the user and a timezone can be
    # found via contractdtails, then try to get it from pytz, which may or
    # may not be available.
    tzstr = isinstance(tz, string_types)
    if tz is None or not tzstr:
        return Localizer(tz)

    try:
        import pytz  # keep the import very local
    except ImportError:
        return Localizer(tz)  # nothing can be done

    tzs = tz
    if tzs == 'CST':  # usual alias
        tzs = 'CST6CDT'

    try:
        tz = pytz.timezone(tzs)
    except pytz.UnknownTimeZoneError:
        return Localizer(tz)  # nothing can be done

    return tz


def Localizer(tz):
    import types

    def localize(self, dt):
        return dt.replace(tzinfo=self)

    if tz is not None and not hasattr(tz, 'localize'):
        # patch the tz instance with a bound method
        tz.localize = types.MethodType(localize, tz)

    return tz


# A UTC class, same as the one in the Python Docs
class _UTC(datetime.tzinfo):
    """UTC"""

    def utcoffset(self, dt):
        return ZERO

    def tzname(self, dt):
        return "UTC"

    def dst(self, dt):
        return ZERO

    def localize(self, dt):
        return dt.replace(tzinfo=self)


class _LocalTimezone(datetime.tzinfo):

    def utcoffset(self, dt):
        if self._isdst(dt):
            return DSTOFFSET
        else:
            return STDOFFSET

    def dst(self, dt):
        if self._isdst(dt):
            return DSTDIFF
        else:
            return ZERO

    def tzname(self, dt):
        return _time.tzname[self._isdst(dt)]

    def _isdst(self, dt):
        tt = (dt.year, dt.month, dt.day,
              dt.hour, dt.minute, dt.second,
              dt.weekday(), 0, 0)
        try:
            stamp = _time.mktime(tt)
        except (ValueError, OverflowError):
            return False  # Too far in the future, not relevant

        tt = _time.localtime(stamp)
        return tt.tm_isdst > 0

    def localize(self, dt):
        return dt.replace(tzinfo=self)


UTC = _UTC()
TZLocal = _LocalTimezone()

HOURS_PER_DAY = 24.0
MINUTES_PER_HOUR = 60.0
SECONDS_PER_MINUTE = 60.0
MUSECONDS_PER_SECOND = 1e6
MINUTES_PER_DAY = MINUTES_PER_HOUR * HOURS_PER_DAY
SECONDS_PER_DAY = SECONDS_PER_MINUTE * MINUTES_PER_DAY
MUSECONDS_PER_DAY = MUSECONDS_PER_SECOND * SECONDS_PER_DAY


def num2date(x, tz=None, naive=True):
    # Same as matplotlib except if tz is None a naive datetime object
    # will be returned.
    """
    *x* is a float value which gives the number of days
    (fraction part represents hours, minutes, seconds) since
    0001-01-01 00:00:00 UTC *plus* *one*.
    The addition of one here is a historical artifact.  Also, note
    that the Gregorian calendar is assumed; this is not universal
    practice.  For details, see the module docstring.
    Return value is a :class:`datetime` instance in timezone *tz* (default to
    rcparams TZ value).
    If *x* is a sequence, a sequence of :class:`datetime` objects will
    be returned.
    """

    ix = int(x)
    dt = datetime.datetime.fromordinal(ix)
    remainder = float(x) - ix

    remainder *= HOURS_PER_DAY
    hour = math.floor(remainder)

    remainder = (remainder - hour) * MINUTES_PER_HOUR
    minute = math.floor(remainder)

    remainder = (remainder - minute) * SECONDS_PER_MINUTE
    second = math.floor(remainder)

    remainder = (remainder - second) * MUSECONDS_PER_SECOND
    microsecond = math.floor(remainder)
    if microsecond < 10:
        microsecond = 0  # compensate for rounding errors

    if tz is not None:
        dt = datetime.datetime(
            dt.year, dt.month, dt.day, int(hour), int(minute), int(second),
            microsecond, tzinfo=UTC)
        dt = dt.astimezone(tz)
        if naive:
            dt = dt.replace(tzinfo=None)
    else:
        # If not tz has been passed return a non-timezoned dt
        dt = datetime.datetime(
            dt.year, dt.month, dt.day, int(hour), int(minute), int(second),
            microsecond)

    if microsecond > 999990:  # compensate for rounding errors
        dt += datetime.timedelta(microseconds=1e6 - microsecond)

    return dt


def num2dt(num, tz=None, naive=True):
    return num2date(num, tz=tz, naive=naive).date()


def num2time(num, tz=None, naive=True):
    return num2date(num, tz=tz, naive=naive).time()


def date2num(dt, tz=None):
    """
    Convert :mod:`datetime` to the Gregorian date as UTC float days,
    preserving hours, minutes, seconds and microseconds.  Return value
    is a :func:`float`.
    1. 先給日期切換好時區
    2. 將日期中的數字部分轉換到標準時間，因為 toordinal 函數與時區無關
    3. 日期部分通過 toordinal 函數實現，時分秒等通過算術運算即可獲得，最後相加即可
    """
    if tz is not None:
        dt = tz.localize(dt)

    # 因為下文中的 toordinal 函數與時區無關，所以這裡時間上要切換到 UTC 時間
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        delta = dt.tzinfo.utcoffset(dt)
        if delta is not None:
            dt -= delta

    base = float(dt.toordinal())
    if hasattr(dt, 'hour'):
        # base += (dt.hour / HOURS_PER_DAY +
        #          dt.minute / MINUTES_PER_DAY +
        #          dt.second / SECONDS_PER_DAY +
        #          dt.microsecond / MUSECONDS_PER_DAY
        #         )
        base = math.fsum(
            (base, dt.hour / HOURS_PER_DAY, dt.minute / MINUTES_PER_DAY,
             dt.second / SECONDS_PER_DAY, dt.microsecond / MUSECONDS_PER_DAY))

    return base


def time2num(tm):
    """
    Converts the hour/minute/second/microsecond part of tm (datetime.datetime
    or time) to a num
    """
    num = (tm.hour / HOURS_PER_DAY +
           tm.minute / MINUTES_PER_DAY +
           tm.second / SECONDS_PER_DAY +
           tm.microsecond / MUSECONDS_PER_DAY)

    return num


def str2dt(dt_string):
    dt_str = dt_string.split('+')[0].split('T')

    date_str = dt_str[0].split('-')
    date_int = [int(s) for s in date_str]

    dt_str_split = dt_str[1].split('.')
    if len(dt_str_split) == 2:
        time_str = dt_str_split[0].split(':') + [dt_str_split[1][:6].ljust(6, "0")]
        time_int = [int(s) for s in time_str]

        dt = datetime.datetime(date_int[0], date_int[1], date_int[2],
                               time_int[0], time_int[1], time_int[2], time_int[3])
        return dt
    else:
        time_str = dt_str_split[0].split(':')
        time_int = [int(s) for s in time_str]

        dt = datetime.datetime(date_int[0], date_int[1], date_int[2],
                               time_int[0], time_int[1], time_int[2])
        return dt


TD1 = datetime.timedelta(days=1)
"""One day of time delta"""


def get_fx_eod_date(dt: datetime):
    """
    Gets EOD date in FX market.
    The date line is set as 5 p.m. N.Y. time./\n
    :param dt: DateTime in Taipei time
    :return: EOD date in FX market.
    """
    dt_us = TZ_TP.localize(dt).astimezone(TZ_US)
    dt = dt_us.date()
    if dt_us.hour >= 17:
        # Switch date after 5 p.m. N.Y. time
        dt = dt + TD1

    return dt
