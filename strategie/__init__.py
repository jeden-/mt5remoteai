"""
Pakiet zawierający strategie tradingowe systemu NikkeiNinja.
"""

from .wyckoff import WyckoffAnalyzer
from .rozpoznawanie_wzorcow import RozpoznawanieWzorcow

__all__ = ['WyckoffAnalyzer', 'RozpoznawanieWzorcow'] 