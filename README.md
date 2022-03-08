MyBackTrader
==========

1. This version is extended from the original backtrader `mementum/backtrader <https://github.com/mementum/backtrader>`_.
2. This strategies engine or so-called back-testing engine is mainly provided for our Quant Traders. This engine communities with our low-latency intelligent order router, 
   which is responsible for taking care of all streaming ticks from APIs.
3. We will add some core features which are listed below.
  * Easy use **ADF test**, **Hurst Exponent**, and **Variance Ratio test** for mean reversion strategies.
  * Easy use **Co-integrated ADF test** and **Johansen test** to find the co-integration.
  * Easy use above statistics test to find intraday momentum strategies.
  * Portfolio Management: 
     - Easy use **Kelly formula** to find optimal leverage for risk management.
     
Features:
=========

原有特色可以參照原版

  - 

Python 3 Support
==================

  - Python = ``3.9``

Version numbering
=================

X.Y.Z.I

  - X: Major version number. Should stay stable unless something big is changed
    like an overhaul to use ``numpy``
  - Y: Minor version number. To be changed upon adding a complete new feature or
    (god forbids) an incompatible API change.
  - Z: Revision version number. To be changed for documentation updates, small
    changes, small bug fixes
  - I: Number of Indicators already built into the platform
