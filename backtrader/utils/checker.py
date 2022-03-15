import datetime


def check_trade_hour(dt: datetime):
    """
    Determines if the datetime is in trade hour.
    Only accept from 6 a.m. on Monday to 6 a.m. on Saturday in Taipei time.
    :param dt: Datetime
    :return: Boolean status whether the datetime is in trade hour.
    """
    # Check if data is in market hour
    if dt.weekday() == 6:
        # Sunday
        return False
    elif dt.weekday() == 5 and dt.hour > 6:
        # Saturday
        return False
    elif dt.weekday() == 0 and dt.hour < 6:
        # Monday
        return False
    return True
