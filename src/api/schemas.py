from typing import List
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    league: str = Field(..., description="League name exactly as in dataset (e.g. 'Premier League')")
    home_team: str
    away_team: str


class MarketProbability(BaseModel):
    market: str
    probability: float
    probability_percent: float
    fair_odds: float | None
    risk_color: str


class PredictResponse(BaseModel):
    league: str
    home_team: str
    away_team: str
    markets: List[MarketProbability]


class LeagueResponse(BaseModel):
    leagues: List[str]


class TeamsResponse(BaseModel):
    league: str
    teams: List[str]
