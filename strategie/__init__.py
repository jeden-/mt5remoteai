"""
Pakiet zawierający strategie tradingowe systemu NikkeiNinja.
"""

from .wyckoff import StrategiaWyckoff
from .rozpoznawanie_wzorcow import RozpoznawanieWzorcow

__all__ = ['StrategiaWyckoff', 'RozpoznawanieWzorcow'] 