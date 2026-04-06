# Türkçe cümle düzeyi duygu analizi

BERT tabanlı **tek etiketli cümle sınıflandırması** (Negative / Neutral / Positive). Ham veri ABSA formatında da olabilir; ön işlemede aynı cümleye ait satırlar **Polarity çoğunluk oyu** ile tek satıra indirgenir.

## Klasör yapısı

| Yol | Açıklama |
|-----|----------|
| `data/` | Ham CSV, train/val/test split, `outputs/` altında toplu tahmin ve grafikler |
| `data/turkish_absa_train.csv` | `data_download.py` çıktısı (Sentence, Aspect, Polarity) veya uyumlu ham dosya |
| `data/train.csv`, `val.csv`, `test.csv` | Sadece `Sentence`, `Polarity` (ön işleme sonrası) |
| `data/hard_examples.csv` | İsteğe bağlı zor örnekler (`Sentence`, `Polarity`); eğitimde baskın etiket |
| `data/sample_tweets.csv` | Toplu tahmin için örnek girdi (`text` sütunu) |
| `data/outputs/` | `sentiment_batch_results.csv`, grafikler, confusion matrix |
| `models/sentence_best_model.bin` | Eğitimde kaydedilen `state_dict` (`config.MODEL_PATH`) |
| `src/` | Eğitim, veri, tahmin, metrik |
| `scripts/` | Yardımcı scriptler |
| `backend/` + `frontend/` | FastAPI + statik demo arayüz |

**Not:** Eski `best_model_state.bin` (özel `SentimentClassifier` / `out` katmanı) bu mimariyle **uyumlu değildir**. Cümle modeli için `train.py` ile yeniden eğitim gerekir.

## Kurulum

Sanal ortamda bağımlılıkları yükleyin (`requirements.txt`).

## Çalıştırma sırası

1. **`src/data_download.py`** — HuggingFace’ten ham veriyi `data/turkish_absa_train.csv` olarak indirir (isteğe bağlı; kendi CSV’nizi de koyabilirsiniz).
2. **`src/data_preprocessing.py`** — Cümle düzeyine indirger, `train/val/test` üretir.
3. **`src/train.py`** — Eğitim havuzunu `train.csv` + (varsa) `turkish_absa_train.csv` + Hugging Face alt kümesi (`config.py`: `USE_HF_TRAIN_EXTRA`, `HF_SAMPLE_SIZE`) ile birleştirir; en iyi modeli `models/sentence_best_model.bin` olarak kaydeder (`MODEL_PATH`). İnternet/HF istemezsen `USE_HF_TRAIN_EXTRA = False`; ham ABSA yoksa `MERGE_RAW_ABSA_FOR_TRAIN = False`.
4. **Tahmin** — Etkileşimli cümle: proje kökünde `python predict.py` **veya** `cd src` → `python predict.py`. (`auto_predict.py` yok; eski ABSA akışı kaldırıldı.)
5. **`src/batch_predict.py`** — `data/sample_tweets.csv` → `data/outputs/sentiment_batch_results.csv`.
6. **`src/evaluate_metrics.py`** — Test seti raporu + `data/outputs/confusion_matrix.png`.
7. **`src/visualize_results.py`** — Toplu sonuç histogramu (`data/outputs/chart_sentiment_distribution.png`).

## Web API

Proje kökünden:

```bash
uvicorn backend.main:app --reload --app-dir .
```

Arayüz kök URL’de; sağlık: `GET /api/health`, tahmin: `POST /api/predict` body `{"text": "..."}`.

## Colab tek script

- `scripts/colab_turkish_sentiment_full.py` doğrudan cümle düzeyi sentiment eğitimi yapar.
- `MERGE_HARD_EXAMPLES` varsayılanı açıktır (`1/true/yes/on`); `0/false` verilirse `hard_examples.csv` birleştirilmez.
- `hard_examples.csv` kullanmak için dosyanın Colab tarafında `DATA_DIR` altında bulunması gerekir.
- Not: Bu proje **ABSA inference** yapmaz; ABSA formatlı ham veri yalnızca eğitim havuzunu beslemek için cümle düzeyine indirgenir.

## Zor örnekleri iyileştirme (ironi, nötr konuşma)

1. `data/hard_examples.csv` dosyasına satır ekle: `Sentence`, `Polarity` (0=Negative, 1=Neutral, 2=Positive).
2. Aynı cümle büyük havuzda da geçse **`hard_examples` etiketi baskın** gelir (`training_data._merge_hard_overrides`).
3. `config.py` içinde `MERGE_HARD_EXAMPLES = True` kalsın; Colab scriptinde de varsayılan açıktır.
4. `train.py` (veya Colab script) ile yeniden eğit.
5. Yeni zor cümleler geldikçe dosyayı büyüt; yüzlerce tutarlı örnek genelde belirgin fark yaratır.

## Yardımcı scriptler

- `scripts/inspect_seq_lengths.py` — `train.csv` içinde cümle kelime uzunluğu özeti.
