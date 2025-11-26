from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass
class TeamStatsState:
    """
    Bir takımın o anki güncel istatistiklerini tutar.
    Ayrıca son 5 maç puanını ve detaylı maç geçmişini saklar.
    """
    matches_played: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    btts_count: int = 0
    over_2_5_count: int = 0
    over_3_5_count: int = 0
    ht_over_1_5_count: int = 0
    corners_over_9_5_count: int = 0

    # Son 5 maçın puanları (G=3, B=1, M=0)
    last_5_points: List[int] = field(default_factory=list)

    # YENİ: Son maçların detaylı listesi (Tarih, Rakip, Skor, Sonuç)
    match_history: List[Dict] = field(default_factory=list)

    # --- GÜÇ HESAPLAMASI (Points Per Match) ---
    @property
    def points_per_match(self) -> float:
        """Takımın genel güç seviyesini (0.0 - 3.0 arası) belirler."""
        if self.matches_played == 0:
            return 0.0
        total_points = (self.wins * 3) + (self.draws * 1)
        return total_points / self.matches_played

    @property
    def form_last_5(self) -> float:
        if not self.last_5_points: return 0.0
        # Maksimum puan 15 (5*3). 0-1 arasına normalize et.
        return sum(self.last_5_points) / 15.0

    # Diğer istatistiksel özellikler
    @property
    def goals_for_avg(self) -> float:
        return self.goals_scored / self.matches_played if self.matches_played else 0.0

    @property
    def goals_against_avg(self) -> float:
        return self.goals_conceded / self.matches_played if self.matches_played else 0.0

    @property
    def total_goals_avg(self) -> float:
        return (self.goals_scored + self.goals_conceded) / self.matches_played if self.matches_played else 0.0

    @property
    def btts_rate(self) -> float:
        return self.btts_count / self.matches_played if self.matches_played else 0.0

    @property
    def over_2_5_rate(self) -> float:
        return self.over_2_5_count / self.matches_played if self.matches_played else 0.0

    @property
    def over_3_5_rate(self) -> float:
        return self.over_3_5_count / self.matches_played if self.matches_played else 0.0

    @property
    def ht_over_1_5_rate(self) -> float:
        return self.ht_over_1_5_count / self.matches_played if self.matches_played else 0.0

    @property
    def corners_over_9_5_rate(self) -> float:
        return self.corners_over_9_5_count / self.matches_played if self.matches_played else 0.0


@dataclass
class FeatureEngineer:
    league_encoder: LabelEncoder = field(default_factory=LabelEncoder)
    team_encoder: LabelEncoder = field(default_factory=LabelEncoder)

    # Her takımın SON durumu (Canlı tahmin için gerekli)
    latest_team_stats: Dict[str, TeamStatsState] = field(default_factory=dict)

    league_teams: Dict[str, List[str]] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    fitted: bool = False

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
        """
        Geçmişten günümüze doğru öğrenerek ilerler (Rolling Logic).
        """
        df = df.copy()

        # Tarih sıralaması
        if "Date" in df.columns:
            df["Date"] = pd.to_numeric(df["Date"], errors='coerce')

        # 1) Target kolonlarını ekle
        df = self._add_targets(df)

        # 2) Lig -> takım haritası
        self._build_league_team_map(df)

        # 3) ID Encoderları eğit
        df = self._encode_ids_fit(df)

        # 4) ROLLING FEATURE HESAPLAMA
        X_rows = []
        current_stats: Dict[str, TeamStatsState] = {}

        for idx, row in df.iterrows():
            home = str(row["HomeTeam"])
            away = str(row["AwayTeam"])

            if home not in current_stats: current_stats[home] = TeamStatsState()
            if away not in current_stats: current_stats[away] = TeamStatsState()

            h_stats = current_stats[home]
            a_stats = current_stats[away]

            features = {
                "league_id": row["league_id"],
                "home_team_id": row["home_team_id"],
                "away_team_id": row["away_team_id"],

                "home_goals_for_avg": h_stats.goals_for_avg,
                "home_goals_against_avg": h_stats.goals_against_avg,
                "home_total_goals_avg": h_stats.total_goals_avg,
                "home_btts_rate": h_stats.btts_rate,
                "home_over_2_5_rate": h_stats.over_2_5_rate,
                "home_over_3_5_rate": h_stats.over_3_5_rate,
                "home_ht_over_1_5_rate": h_stats.ht_over_1_5_rate,
                "home_corners_over_9_5_rate": h_stats.corners_over_9_5_rate,
                "home_form_last_5": h_stats.form_last_5,

                "away_goals_for_avg": a_stats.goals_for_avg,
                "away_goals_against_avg": a_stats.goals_against_avg,
                "away_total_goals_avg": a_stats.total_goals_avg,
                "away_btts_rate": a_stats.btts_rate,
                "away_over_2_5_rate": a_stats.over_2_5_rate,
                "away_over_3_5_rate": a_stats.over_3_5_rate,
                "away_ht_over_1_5_rate": a_stats.ht_over_1_5_rate,
                "away_corners_over_9_5_rate": a_stats.corners_over_9_5_rate,
                "away_form_last_5": a_stats.form_last_5,
            }
            X_rows.append(features)

            self._update_stats(current_stats[home], row, is_home=True)
            self._update_stats(current_stats[away], row, is_home=False)

        # Döngü bittiğinde son durumu kaydet
        self.latest_team_stats = current_stats

        X = pd.DataFrame(X_rows)
        self.feature_columns = list(X.columns)

        targets = {
            "result_1x2": df["result_1x2"],
            "btts_yes": df["btts_yes"],
            "over_2_5": df["over_2_5"],
            "over_3_5": df["over_3_5"],
            "first_half_over_1_5": df["first_half_over_1_5"],
            "corners_over_9_5": df["corners_over_9_5"],
        }

        self.fitted = True
        return X, targets

    def transform_match(self, league: str, home_team: str, away_team: str) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FeatureEngineer is not fitted.")

        # Takım verisi yoksa boş statü oluştur
        h_stats = self.latest_team_stats.get(home_team, TeamStatsState())
        a_stats = self.latest_team_stats.get(away_team, TeamStatsState())

        try:
            l_id = int(self.league_encoder.transform([league])[0])
            h_id = int(self.team_encoder.transform([home_team])[0])
            a_id = int(self.team_encoder.transform([away_team])[0])
        except:
            l_id, h_id, a_id = -1, -1, -1

        data = {
            "league_id": l_id,
            "home_team_id": h_id,
            "away_team_id": a_id,

            "home_goals_for_avg": h_stats.goals_for_avg,
            "home_goals_against_avg": h_stats.goals_against_avg,
            "home_total_goals_avg": h_stats.total_goals_avg,
            "home_btts_rate": h_stats.btts_rate,
            "home_over_2_5_rate": h_stats.over_2_5_rate,
            "home_over_3_5_rate": h_stats.over_3_5_rate,
            "home_ht_over_1_5_rate": h_stats.ht_over_1_5_rate,
            "home_corners_over_9_5_rate": h_stats.corners_over_9_5_rate,
            "home_form_last_5": h_stats.form_last_5,

            "away_goals_for_avg": a_stats.goals_for_avg,
            "away_goals_against_avg": a_stats.goals_against_avg,
            "away_total_goals_avg": a_stats.total_goals_avg,
            "away_btts_rate": a_stats.btts_rate,
            "away_over_2_5_rate": a_stats.over_2_5_rate,
            "away_over_3_5_rate": a_stats.over_3_5_rate,
            "away_ht_over_1_5_rate": a_stats.ht_over_1_5_rate,
            "away_corners_over_9_5_rate": a_stats.corners_over_9_5_rate,
            "away_form_last_5": a_stats.form_last_5,
        }

        X = pd.DataFrame([data])
        X = X.reindex(columns=self.feature_columns, fill_value=0.0)
        return X

    def _update_stats(self, stats: TeamStatsState, row: pd.Series, is_home: bool):
        fthg = row["FTHG"]
        ftag = row["FTAG"]

        my_goals = fthg if is_home else ftag
        opp_goals = ftag if is_home else fthg

        # Detaylı bilgi için rakip ve tarih
        opponent = row["AwayTeam"] if is_home else row["HomeTeam"]
        date_val = row.get("Date", None)

        stats.matches_played += 1
        stats.goals_scored += my_goals
        stats.goals_conceded += opp_goals

        # Form Puanı ve Sonuç Karakteri (W, D, L)
        match_points = 0
        result_char = "D"

        if my_goals > opp_goals:
            stats.wins += 1;
            match_points = 3;
            result_char = "W"
        elif my_goals == opp_goals:
            stats.draws += 1;
            match_points = 1;
            result_char = "D"
        else:
            stats.losses += 1;
            match_points = 0;
            result_char = "L"

        # Son 5 maçın puanını güncelle
        stats.last_5_points.append(match_points)
        if len(stats.last_5_points) > 5: stats.last_5_points.pop(0)

        # --- MAÇ GEÇMİŞİNİ KAYDET (KRİTİK KISIM) ---
        match_info = {
            "opponent": opponent,
            "score": f"{my_goals} - {opp_goals}",
            "result": result_char,  # W, D, L
            "date": str(date_val).split(" ")[0] if date_val else "-"
        }

        # Listeye ekle (Sadece son 10 maçı tutalım ki hafıza şişmesin, bize 5 lazım ama 10 tutmak güvenli)
        stats.match_history.insert(0, match_info)  # En yeni maç en başa
        if len(stats.match_history) > 10:
            stats.match_history.pop()
        # ------------------------------------------

        if (fthg > 0) and (ftag > 0): stats.btts_count += 1
        if (fthg + ftag) > 2.5: stats.over_2_5_count += 1
        if (fthg + ftag) > 3.5: stats.over_3_5_count += 1

        hthg = row.get("HTHG", 0)
        htag = row.get("HTAG", 0)
        if (hthg + htag) > 1.5: stats.ht_over_1_5_count += 1

        hc = row.get("HC", 0)
        ac = row.get("AC", 0)
        if (hc + ac) > 9.5: stats.corners_over_9_5_count += 1

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["FTHG", "FTAG"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["FTHG", "FTAG"]).copy()
        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)

        df["result_1x2"] = np.select(
            [df["FTHG"] > df["FTAG"], df["FTHG"] == df["FTAG"]], [1, 0], default=2
        ).astype(int)

        df["btts_yes"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
        total = df["FTHG"] + df["FTAG"]
        df["over_2_5"] = (total > 2.5).astype(int)
        df["over_3_5"] = (total > 3.5).astype(int)

        for col in ["HTHG", "HTAG", "HC", "AC"]:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["first_half_over_1_5"] = ((df["HTHG"] + df["HTAG"]) > 1.5).astype(int)
        df["corners_over_9_5"] = ((df["HC"] + df["AC"]) > 9.5).astype(int)
        return df

    def _build_league_team_map(self, df: pd.DataFrame):
        league_map = {}
        for league in df["league"].unique():
            subset = df[df["league"] == league]
            teams = pd.unique(subset[["HomeTeam", "AwayTeam"]].values.ravel("K"))
            league_map[league] = sorted(list(teams))
        self.league_teams = league_map

    def _encode_ids_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        leagues = df["league"].astype(str)
        self.league_encoder.fit(leagues)
        df["league_id"] = self.league_encoder.transform(leagues)

        teams_all = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel("K").astype(str))
        self.team_encoder.fit(teams_all)
        df["home_team_id"] = self.team_encoder.transform(df["HomeTeam"].astype(str))
        df["away_team_id"] = self.team_encoder.transform(df["AwayTeam"].astype(str))
        return df