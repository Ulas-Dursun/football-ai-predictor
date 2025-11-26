import os
import shutil
from pathlib import Path
from  src.application.model_training import train_models


def clean_models():
    root = Path(".").resolve()
    model_dir = root / "models"

    print(f"🧹 Temizlik Başlatılıyor: {model_dir}")

    if model_dir.exists():
        # Klasörün içindeki tüm .pkl dosyalarını sil
        count = 0
        for file in model_dir.glob("*.pkl"):
            try:
                file.unlink()  # Dosyayı sil
                count += 1
                print(f"   🗑️ Silindi: {file.name}")
            except Exception as e:
                print(f"   ❌ Silinemedi {file.name}: {e}")

        if count == 0:
            print("   ℹ️ Klasör zaten boş.")
        else:
            print(f"   ✅ Toplam {count} model dosyası temizlendi.")
    else:
        print("   ℹ️ 'models' klasörü bulunamadı, eğitim sırasında oluşturulacak.")

    print("\n" + "=" * 40)
    print("🏋️ SIFIRDAN EĞİTİM BAŞLIYOR...")
    print("=" * 40 + "\n")

    # Mevcut eğitim fonksiyonunu çağır
    try:
        train_models()
        print("\n" + "=" * 40)
        print("🎉 İŞLEM TAMAMLANDI! Modellerin artık tertemiz.")
        print("Lütfen web sitesini yeniden başlat: uvicorn src.api.main:app --reload")
        print("=" * 40)
    except Exception as e:
        print(f"\n❌ Eğitim sırasında hata oluştu: {e}")


if __name__ == "__main__":
    clean_models()