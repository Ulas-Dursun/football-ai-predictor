from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from xgboost import XGBClassifier  # YENİ MODEL
from sklearn.calibration import CalibratedClassifierCV  # YENİ KALİBRASYON
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

from src.infrastructure.repositories import CsvMatchRepository
from src.application.feature_engineering import FeatureEngineer


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def train_models() -> None:
    root = get_project_root()
    data_root = root / "data" / "raw"

    # LİG TANIMLAMALARI
    league_dirs = {
        "Premier League": "football_data/premier_league",
        "La Liga": "football_data/la_liga",
        "Serie A": "football_data/serie_a",
        "Bundesliga": "football_data/bundesliga",
        "Super Lig": "football_data/super_lig",
        "Champions League": "ucl"
    }

    # --- DEBUG: KLASÖR KONTROLÜ ---
    print("-" * 40)
    print(f"[DEBUG] Veri Kök Dizini: {data_root}")
    ucl_path = data_root / "ucl"
    if ucl_path.exists():
        csv_files = list(ucl_path.glob("*.csv"))
        if csv_files:
            print(f"[DEBUG] ✅ UCL Klasörü bulundu ve dolu.")
        else:
            print(f"[DEBUG] ❌ UCL Klasörü BOŞ! Lütfen .csv dosyasını buraya at.")
    else:
        print(f"[DEBUG] ❌ UCL Klasörü BULUNAMADI! Aranan yol: {ucl_path}")
    print("-" * 40)
    # ------------------------------

    repo = CsvMatchRepository(
        data_root=data_root,
        league_dirs=league_dirs
    )

    print("[TRAIN] Loading raw data...")
    df = repo.load_all()
    print(f"[TRAIN] Loaded {len(df)} matches total.")

    ucl_matches = len(df[df["league"] == "Champions League"])
    print(f"🏆 Yüklenen Şampiyonlar Ligi maçı: {ucl_matches}")

    # --- TARİH SIRALAMASI ---
    print("[TRAIN] Sorting data by date...")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
        df = df.sort_values("Date").reset_index(drop=True)
        df = df.dropna(subset=["Date"])

    print(f"[TRAIN] Processing features (Rolling Stats + Form Last 5)...")

    fe = FeatureEngineer()
    X, targets = fe.fit_transform(df)

    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(fe, model_dir / "feature_engineer.pkl")
    print("[TRAIN] Saved feature_engineer.pkl")

    configs: Dict[str, str] = {
        "result_1x2": "result_1x2_model.pkl",
        "btts_yes": "btts_model.pkl",
        "over_2_5": "over25_model.pkl",
        "over_3_5": "over35_model.pkl",
        "first_half_over_1_5": "ht_over15_model.pkl",
        "corners_over_9_5": "corners_over95_model.pkl",
    }

    print("\n🚀 XGBoost Eğitimi Başlıyor (Kalibrasyonlu)...")

    for target_name, filename in configs.items():
        y = targets[target_name]
        print(f"   --> Training {target_name}...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 1. Temel XGBoost Modeli
        base_clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42
        )

        # 2. Olasılık Kalibrasyonu (Isotonic Regression)
        # Bu işlem modelin verdiği %80'in gerçekten %80 ihtimal olmasını sağlar.
        calibrated_clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=3)

        # Modeli Eğit
        calibrated_clf.fit(X_train, y_train)

        # Test Et
        y_pred = calibrated_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Binary ise Log Loss da yazdıralım
        try:
            y_prob = calibrated_clf.predict_proba(X_test)
            loss = log_loss(y_test, y_prob)
            print(f"       Accuracy: {acc:.3f} | LogLoss: {loss:.3f}")
        except:
            print(f"       Accuracy: {acc:.3f}")

        joblib.dump(calibrated_clf, model_dir / filename)

    print("\n✅ TÜM MODELLER EĞİTİLDİ VE KAYDEDİLDİ.")


if __name__ == "__main__":
    train_models()