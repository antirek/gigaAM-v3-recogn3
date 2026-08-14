# GigaAM + Sortformer (offline, 2 speakers)

Распознавание моно-записи разговора: **Sortformer** (диаризация) + **GigaAM v3_e2e_rnnt** (ASR).

## Требования

- Docker + NVIDIA Container Toolkit
- GPU с поддержкой **sm_120** (RTX 50xx) — в образах PyTorch **cu128**
- `ffmpeg`, Python 3.8+ на хосте (оркестратор без тяжёлых зависимостей)
- Опционально: `HF_TOKEN` для скачивания Sortformer с Hugging Face

## Быстрый старт

```bash
cp .env.example .env   # при необходимости добавьте HF_TOKEN=...

# Сборка образов (долго: особенно diar / NeMo)
docker-compose build

# Прогон
python3 recognize.py --audio data/mono3.wav --out out/mono3
```

Результат: `out/mono3/transcript.txt`

```text
[00:12] Спикер 1: ...
[00:18] Спикер 2: ...
```

Также: `diar.raw.json`, `segments.json`, `transcript.json`.

## Архитектура

| Сервис | Образ | Роль |
|--------|-------|------|
| `diar` | NeMo + Sortformer | сегменты спикеров |
| `asr` | GigaAM `v3_e2e_rnnt` | текст по сегментам |
| host CLI | `recognize.py` | merge, нарезка, TXT |

Контейнеры запускаются **последовательно** (сначала diar, потом asr).

## Тестовые файлы

- `data/mono1.wav` (~12 мин)
- `data/mono2.wav` (~6 мин)
- `data/mono3.wav` (~2.5 мин) — для первого smoke

## Тюнинг

В `.env` / окружении:

- `AUDIO_ENHANCE` — `highpass` (по умолчанию), `false`, или `true` (highpass+loudnorm; loudnorm на mono3 портил diar)
- `AUDIO_HIGHPASS_HZ` — default `80`
- `MERGE_GAP_SEC` — склейка соседних реплик одного спикера (default `1.2`)
- `MIN_SEGMENT_SEC` (default `0.35`)
- `MAX_ASR_SEC` — нарезка длинных реплик по паузам diar (default `20`)
- `DIAR_MODEL` — по умолчанию `nvidia/diar_streaming_sortformer_4spk-v2.1`
- `DIAR_CHUNK_LEN` и др. — streaming-конфиг Sortformer (very high latency)

## Кэш моделей

- GigaAM: `data/models/gigaam/`
- HuggingFace / NeMo: `data/models/hf`, `data/models/nemo`
