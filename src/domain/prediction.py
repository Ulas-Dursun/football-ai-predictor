from dataclasses import dataclass

@dataclass
class MarketPrediction:
    market: str
    probability: float
    probability_percent: float
    fair_odds: float
    risk_color: str

@dataclass
class BetSuggestion:
    title: str
    market_name: str
    odds: float
    stake_percent: int
    stake_amount: int
    risk_class: str