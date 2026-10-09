"""Final Freeze Paketi, Adim 1: P1 madde 11'in tek seferlik/manuel
adimini (bkz. REPRODUCE.md "Reprodüksiyon açığı" notu) kalici, tekrar
calistirilabilir bir script'e donusturur.

Madde 11'in orijinal kararı (bkz. `06_MODEL_SECIM_RAPORU_PAH.md`, "P1
Madde 11: Provenance-Riskli Özelliklerin Çıkarılması"): `al_all_missing`
ve `CAT_1`, kaynak-provenance kısayolu riski tasidigi icin (ablasyon
denemesi maliyetsiz oldugunu gosterdi -- AUPRC farki p=0,83) 28-ozellik
final havuzundan elle cikarilmisti. Bu script o cikarma islemini,
`f0_final_feature_pool.json`'dan (Adim 20'nin ciktisi, 28 ozellik)
`f0_final_feature_pool_v2.json`'a (26 ozellik) giden yolu YENIDEN
URETIR -- hicbir model/ozellik SECIMI yeniden calistirilmiyor (Adim 1 --
ozellik secimi -- bu turda da tekrarlanmiyor, tipki orijinal manuel
adimda oldugu gibi), yalnizca sabit iki ismin cikarilmasi otomatiklestiriliyor.

Calistirma: python -m genova.pah.f0_madde11_pool_reduction
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
SOURCE_POOL_PATH = TAB_DIR / "f0_final_feature_pool.json"
OUT_POOL_PATH = TAB_DIR / "f0_final_feature_pool_v2.json"

REMOVED_FOR_PROVENANCE_RISK = ["al_all_missing", "CAT_1"]
RULE_SUFFIX = (" -- P1 madde 11: al_all_missing/CAT_1 provenance-riski nedeniyle "
               "elle cikarildi (denetim ablasyonu: AUPRC farki p=0.83, maliyetsiz)")


def reduce_pool(source):
    features = [f for f in source["features"] if f not in REMOVED_FOR_PROVENANCE_RISK]
    missing = [f for f in REMOVED_FOR_PROVENANCE_RISK if f not in source["features"]]
    if missing:
        raise ValueError(f"kaynak havuzda beklenen ama bulunamayan ozellik(ler): {missing}")
    return {
        "rule": source["rule"] + RULE_SUFFIX,
        "n_features": len(features),
        "features": features,
        "removed_for_provenance_risk": REMOVED_FOR_PROVENANCE_RISK,
    }


def main():
    source = json.loads(SOURCE_POOL_PATH.read_text(encoding="utf-8"))
    print(f"kaynak havuz: {SOURCE_POOL_PATH.name} ({source['n_features']} ozellik)", flush=True)

    reduced = reduce_pool(source)
    print(f"indirgenmis havuz: {reduced['n_features']} ozellik "
          f"(cikarilan: {REMOVED_FOR_PROVENANCE_RISK})", flush=True)

    if OUT_POOL_PATH.exists():
        existing = json.loads(OUT_POOL_PATH.read_text(encoding="utf-8"))
        if existing == reduced:
            print(f"DOGRULANDI: uretilen icerik, mevcut {OUT_POOL_PATH.name} ile BIREBIR AYNI.", flush=True)
        else:
            print(f"UYARI: uretilen icerik, mevcut {OUT_POOL_PATH.name}'dan FARKLI -- "
                  f"dosyaya YAZILMADI, DURDURULDU.", flush=True)
            print(f"  mevcut: {existing}", flush=True)
            print(f"  uretilen: {reduced}", flush=True)
            return
    else:
        OUT_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_POOL_PATH.write_text(json.dumps(reduced, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"YAZILDI (dosya yoktu): {OUT_POOL_PATH}", flush=True)


if __name__ == "__main__":
    main()
