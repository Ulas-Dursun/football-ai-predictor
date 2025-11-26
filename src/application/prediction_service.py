from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import joblib
import numpy as np

# Gerekli importlar
from src.domain.match import MatchInput
from src.domain.prediction import MarketPrediction, BetSuggestion
from src.application.feature_engineering import FeatureEngineer


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class PredictionService:
    def __init__(self):
        root = get_project_root()
        model_dir = root / "models"

        # FeatureEngineer'ı yükle
        self.feature_engineer: FeatureEngineer = joblib.load(model_dir / "feature_engineer.pkl")
        print("✅ PredictionService: Model ve İstatistikler Yüklendi.")

        # Modelleri yükle (XGBoost / Calibrated)
        self.models = {
            "result_1x2": joblib.load(model_dir / "result_1x2_model.pkl"),
            "btts_yes": joblib.load(model_dir / "btts_model.pkl"),
            "over_2_5": joblib.load(model_dir / "over25_model.pkl"),
            "over_3_5": joblib.load(model_dir / "over35_model.pkl"),
            "first_half_over_1_5": joblib.load(model_dir / "ht_over15_model.pkl"),
            "corners_over_9_5": joblib.load(model_dir / "corners_over95_model.pkl"),
        }

    def get_leagues(self) -> List[str]:
        return list(self.feature_engineer.league_teams.keys())

    def get_teams_for_league(self, league: str) -> List[str]:
        return self.feature_engineer.league_teams.get(league, [])

    def predict_for_match(self, match: MatchInput) -> Dict:
        """
        Bir maç için tahminleri, takım formlarını ve benzer maç geçmişini üretir.
        """
        # 1. Feature Engineering (Maç için sayısal verileri hazırla)
        X = self.feature_engineer.transform_match(
            league=match.league,
            home_team=match.home_team,
            away_team=match.away_team,
        )

        # 2. İstatistikleri Çek (Güç, Form, Geçmiş)
        h_stats = self.feature_engineer.latest_team_stats.get(match.home_team)
        a_stats = self.feature_engineer.latest_team_stats.get(match.away_team)

        home_power = h_stats.points_per_match if h_stats else 0.0
        away_power = a_stats.points_per_match if a_stats else 0.0

        home_form = h_stats.last_5_points if h_stats else []
        away_form = a_stats.last_5_points if a_stats else []

        # DEBUG: Analiz edilen maçı yazdır
        print(f"\n🔍 ANALİZ: {match.home_team} (Güç: {home_power:.2f}) vs {match.away_team} (Güç: {away_power:.2f})")

        # 3. Benzer Maçları Getir (Filtreleme)
        # Ev sahibi için: Rakibi (Deplasman) gücündeki takımlarla maçları
        home_history_filtered = self._get_similar_matches(match.home_team, target_strength=away_power)

        # Deplasman için: Rakibi (Ev Sahibi) gücündeki takımlarla maçları
        away_history_filtered = self._get_similar_matches(match.away_team, target_strength=home_power)

        # 4. Model Tahminleri
        preds: List[MarketPrediction] = []

        # --- 1X2 Tahmini ---
        clf_1x2 = self.models["result_1x2"]
        proba_1x2 = clf_1x2.predict_proba(X)[0]
        classes = clf_1x2.classes_
        probs = {int(c): p for c, p in zip(classes, proba_1x2)}
        p1, px, p2 = probs.get(1, 0.0), probs.get(0, 0.0), probs.get(2, 0.0)

        preds.append(self._make_pred("Full Time 1", p1))
        preds.append(self._make_pred("Full Time X", px))
        preds.append(self._make_pred("Full Time 2", p2))

        # --- Çifte Şans & DNB ---
        preds.append(self._make_pred("Double Chance 1X", p1 + px))
        preds.append(self._make_pred("Double Chance X2", px + p2))
        preds.append(self._make_pred("Double Chance 12", p1 + p2))

        denom = p1 + p2 if (p1 + p2) > 0 else 1.0
        preds.append(self._make_pred("Home DNB", p1 / denom))
        preds.append(self._make_pred("Away DNB", p2 / denom))

        # --- Binary Modeller ---
        p_btts = self._predict_binary("btts_yes", X)
        preds.append(self._make_pred("BTTS Yes", p_btts))
        preds.append(self._make_pred("BTTS No", 1.0 - p_btts))

        p_o25 = self._predict_binary("over_2_5", X)
        preds.append(self._make_pred("Over 2.5", p_o25))
        preds.append(self._make_pred("Under 2.5", 1.0 - p_o25))

        p_o35 = self._predict_binary("over_3_5", X)
        preds.append(self._make_pred("Over 3.5", p_o35))
        preds.append(self._make_pred("Under 3.5", 1.0 - p_o35))

        p_ht = self._predict_binary("first_half_over_1_5", X)
        preds.append(self._make_pred("1st Half Over 1.5", p_ht))
        preds.append(self._make_pred("1st Half Under 1.5", 1.0 - p_ht))

        p_cor = self._predict_binary("corners_over_9_5", X)
        preds.append(self._make_pred("Corners Over 9.5", p_cor))
        preds.append(self._make_pred("Corners Under 9.5", 1.0 - p_cor))

        # Sonuçları Paketle
        return {
            "predictions": preds,
            "home_form": home_form,
            "away_form": away_form,
            "home_history": home_history_filtered,
            "away_history": away_history_filtered
        }

    def generate_suggestions(self, preds: List[MarketPrediction]) -> List[BetSuggestion]:
        """Yapay zeka destekli kupon önerileri."""
        suggestions = []
        if not preds: return suggestions

        sorted_preds = sorted(preds, key=lambda x: x.probability, reverse=True)

        # 1. GÜVENLİ (Safe) - %70 üstü
        safe_bet = next((p for p in sorted_preds if p.probability >= 0.70), None)
        # Eğer güvenli yoksa en yüksek olasılığı al (Fallback)
        if not safe_bet and sorted_preds: safe_bet = sorted_preds[0]

        if safe_bet:
            suggestions.append(BetSuggestion(
                title="🛡️ Güvenli Tercih",
                market_name=safe_bet.market,
                odds=safe_bet.fair_odds,
                stake_percent=5, stake_amount=50, risk_class="light_green"
            ))

        # 2. İDEAL (Value) - %55-%70 arası
        ideal_bet = next((p for p in sorted_preds if 0.55 <= p.probability < 0.70), None)
        if ideal_bet:
            suggestions.append(BetSuggestion(
                title="⚖️ İdeal / Değer",
                market_name=ideal_bet.market,
                odds=ideal_bet.fair_odds,
                stake_percent=3, stake_amount=30, risk_class="yellow"
            ))

        # 3. SÜRPRİZ (High Risk) - %35-%55 arası
        risky_bet = next((p for p in sorted_preds if 0.35 <= p.probability < 0.55), None)
        if risky_bet:
            suggestions.append(BetSuggestion(
                title="🔥 Yüksek Getiri",
                market_name=risky_bet.market,
                odds=risky_bet.fair_odds,
                stake_percent=1, stake_amount=10, risk_class="red"
            ))

        return suggestions

    def _get_similar_matches(self, team_name: str, target_strength: float, tolerance: float = 0.5) -> List[Dict]:
        """
        Benzer güçteki rakipleri arar, bulamazsa kesinlikle son 5 maçı döner.
        """
        stats = self.feature_engineer.latest_team_stats.get(team_name)

        # Takım yoksa veya geçmişi boşsa boş dön
        if not stats or not stats.match_history:
            print(f"⚠️ [DEBUG] {team_name} geçmiş verisi bulunamadı.")
            return []

        # İç Fonksiyon: Filtreleme
        def filter_by_tolerance(tol):
            filtered = []
            for match in stats.match_history:
                opp_name = match["opponent"]
                opp_stats = self.feature_engineer.latest_team_stats.get(opp_name)
                opp_strength = opp_stats.points_per_match if opp_stats else 1.3  # Varsayılan güç

                if abs(opp_strength - target_strength) <= tol:
                    m_copy = match.copy()
                    m_copy["opp_strength"] = opp_strength
                    filtered.append(m_copy)
            return filtered

        # 1. DENEME: Dar Tolerans (+/- 0.5 puan)
        matches = filter_by_tolerance(tolerance)

        # 2. DENEME: Geniş Tolerans (+/- 1.0 puan) - Eğer az maç varsa
        if len(matches) < 3:
            matches = filter_by_tolerance(tolerance + 0.5)

        # 3. FALLBACK (Yedek Plan): Son 5 maçı olduğu gibi al
        if not matches:
            print(f"⚠️ [DEBUG] {team_name} için benzer rakip yok, son 5 maç getiriliyor.")
            fallback_matches = []
            for m in stats.match_history[:5]:
                m_copy = m.copy()
                opp_stats = self.feature_engineer.latest_team_stats.get(m["opponent"])
                m_copy["opp_strength"] = opp_stats.points_per_match if opp_stats else 0.0
                fallback_matches.append(m_copy)
            return fallback_matches

        # En fazla 5 maç döndür
        return matches[:5]

    def _make_pred(self, market: str, prob: float) -> MarketPrediction:
        pct = prob * 100
        if pct >= 75:
            color = "dark_green"
        elif pct >= 60:
            color = "light_green"
        elif pct >= 50:
            color = "yellow"
        elif pct >= 35:
            color = "orange"
        else:
            color = "red"

        # Adil oran hesabı (1/olasılık)
        fair_odds = 1.0 / prob if prob > 0.01 else 99.0

        return MarketPrediction(
            market=market,
            probability=prob,
            probability_percent=pct,
            fair_odds=fair_odds,
            risk_color=color
        )

    def _predict_binary(self, key: str, X) -> float:
        clf = self.models[key]
        proba = clf.predict_proba(X)[0]
        classes = clf.classes_
        if len(classes) == 1: return 1.0 if int(classes[0]) == 1 else 0.0

        idx_arr = np.where(classes == 1)[0]
        return float(proba[idx_arr[0]]) if len(idx_arr) > 0 else 0.0