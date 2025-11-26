from dataclasses import dataclass


@dataclass
class MatchInput:
    league: str
    home_team: str
    away_team: str
