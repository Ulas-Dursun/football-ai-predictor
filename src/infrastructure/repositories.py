from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
from io import StringIO


class CsvMatchRepository:
    def __init__(
            self,
            data_root: Path,
            league_dirs: Dict[str, str],
            ucl_dir: Optional[str] = None
    ):
        self.data_root = data_root
        self.league_dirs = league_dirs

    def load_all(self) -> pd.DataFrame:
        all_dfs: List[pd.DataFrame] = []

        print(f"[REPO] Veri okunuyor: {self.data_root}")

        for league_name, folder_name in self.league_dirs.items():
            folder_path = self.data_root / folder_name

            # Eğer klasör bulunamazsa atla
            if not folder_path.exists():
                continue

            # Klasördeki tüm CSV'leri bul
            csv_files = list(folder_path.glob("*.csv"))
            if not csv_files:
                continue

            print(f"[REPO] {league_name} ({folder_name}) yükleniyor...")

            for csv_file in csv_files:
                try:
                    # -------------------------------------------------------
                    # AKILLI OKUMA MANTIĞI (Burayı Senin İçin Entegre Ettim)
                    # -------------------------------------------------------
                    df = self._read_clean_csv(csv_file)

                    if df is not None:
                        # Lig ismini ekle
                        df["league"] = league_name
                        all_dfs.append(df)

                except Exception as e:
                    print(f"[REPO] HATA ({csv_file.name}): {e}")

        if not all_dfs:
            raise RuntimeError("Hiçbir veri yüklenemedi! Klasör yollarını kontrol et.")

        final_df = pd.concat(all_dfs, ignore_index=True)
        return final_df

    def _read_clean_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Dosyayı okur, bozuk satırları () temizler,
        sütun isimlerini standartlaştırır.
        """
        try:
            # 1. Dosyayı metin olarak oku ve temizle
            clean_lines = []
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # Senin dosyadaki bozuk satırları at
                    if "[source" in line or line.strip() == "":
                        continue
                    clean_lines.append(line)

            # 2. Pandas ile yükle
            csv_content = "".join(clean_lines)
            df = pd.read_csv(StringIO(csv_content))

            # 3. Sütun isimlerini temizle ve standartlaştır
            df.columns = [c.strip() for c in df.columns]

            rename_map = {
                "homeTeam": "HomeTeam", "awayteam": "AwayTeam", "awayTeam": "AwayTeam",
                "homeScore": "FTHG", "awayscore": "FTAG", "awayScore": "FTAG",
                "date": "Date"
            }
            df = df.rename(columns=rename_map)

            # 4. Gerekli sütunlar var mı kontrol et
            required = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
            missing = [c for c in required if c not in df.columns]

            if missing:
                # Eğer bu kolonlar yoksa, muhtemelen yanlış bir dosyadır
                print(f"[REPO] UYARI: {file_path.name} dosyasında eksik sütunlar: {missing}")
                return None

            # 5. Tarihi düzelt (Varsa)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                df = df.dropna(subset=["Date"])

            return df

        except Exception as e:
            print(f"[REPO] Kritik Okuma Hatası {file_path.name}: {e}")
            return None