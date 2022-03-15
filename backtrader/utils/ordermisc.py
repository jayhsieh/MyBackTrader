import math

# parameters ------------------------------------------
# slippage
slippage_dict = {'USDMXN': 0.001, 'USDZAR': 0.001, 'USDSEK': 0.001, 'USDSGD': 0.0005}
slippage_basic = 0.0002

unit_basic = 1000  # minimum size of transaction
size_portion = 0.9  # cash portion to trade

# factor
factor_dict = {'USDJPY': 100, 'USDTWD': 10}


# -----------------------------------------------------

def get_factor(ccy: str):
    """
    Gets price factor of given currency pair.
    This is the minimum price change unit for transaction.\n
    :param ccy: Currency pair in XXXYYY.
    :return: Price factor
    """
    ccy = ccy[:6]
    if ccy in factor_dict:
        f = factor_dict[ccy]
    else:
        f = 1

    return 0.00001 * f


def get_bp(ccy: str):
    """
    Gets price basis point of given currency pair.\n
    :param ccy: Currency pair in XXXYYY.
    :return: Price basis point
    """
    ccy = ccy[:6]
    if ccy in factor_dict:
        f = factor_dict[ccy]
    else:
        f = 1

    return 0.0001 * f


def factor_down(price: float, ccy: str):
    """
    Round down price by ``factor`` as unit.\n
    :param price: Price to round down
    :param ccy: Currency pair in XXXYYY
    :return: Rounded price
    """
    factor = get_factor(ccy)
    return math.floor(price / factor) * factor


def factor_up(price: float, ccy: str):
    """
    Round up price by ``factor`` as unit.\n
    :param price: Price to round up
    :param ccy: Currency pair in XXXYYY
    :return: Rounded price
    """
    factor = get_factor(ccy)
    return math.ceil(price / factor) * factor
