# План: офлайн-распознавание разговора (GigaAM) + диаризация (Sortformer)

## Цель

По готовому **моно**-аудиофайлу с **двумя** спикерами получить текст разговора:

```text
[00:12] Спикер 1: ...
[00:18] Спикер 2: ...
```

| Компонент | Модель | Роль |
|-----------|--------|------|
| ASR | **GigaAM** `v3_e2e_rnnt` | распознавание русской речи |
| Диаризация | [nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) | кто говорит и когда |

Режим: **offline по файлу**. Sortformer вызываем через `diarize()` (streaming-модель в batch-режиме на файл).

Проект **самостоятельный**: свой код, зависимости и артефакты.

---

## Зафиксированные решения

| Вопрос | Решение |
|--------|---------|
| Вход | **моно-файл**, 2 голоса в одном канале |
| Тестовые файлы | `data/mono1.wav` (~12:06), `data/mono2.wav` (~6:20), `data/mono3.wav` (~2:26) — уже **16 kHz, mono, PCM s16le** |
| Спикеры | ровно **2**: `Спикер 1` / `Спикер 2` |
| Язык | **русский** (GigaAM) |
| Оценка | **на глаз** по прогонам на `data/mono*.wav` (без автоматической метрики/эталона) |
| Выход (primary) | **TXT** с таймкодами реплик: `[MM:SS] Спикер N: текст` |
| Выход (secondary) | `diar.json` + `transcript.json` для отладки (не обязательны пользователю) |
| Word-level timestamps | **не требуются** (таймкоды на уровне реплик) |
| Склейка ASR ↔ diar | **стратегия C**: diar → merge соседних сегментов одного спикера → ASR на кусках |
| Overlap | **один доминирующий** спикер по длительности пересечения (без дублирования текста) |
| ASR модель | GigaAM **`v3_e2e_rnnt`**, GPU (CUDA) |
| Diar рантайм | **NVIDIA NeMo** (`SortformerEncLabelModel`) — официальный Python API |
| Diar конфиг | **high / very high latency** (offline, приоритет качества над latency) |
| Интерфейс MVP | **CLI** (без UI/API на первом этапе) |
| Зависимости | **два Docker-образа**: `diar` (NeMo+Sortformer) и `asr` (GigaAM); оркестратор снаружи |
| Железо | **RTX 5060 Ti 16 GB** — CUDA; оба сервиса по очереди или с разделением VRAM |

---

## Архитектура

```text
                    host CLI (recognize)
                           │
           ┌───────────────┼───────────────┐
           ▼                               ▼
   docker: diar                     docker: asr
   NeMo + Sortformer                GigaAM v3_e2e_rnnt
   → diar.json                      ← сегменты WAV / пути
           │                               │
           └───────────────┬───────────────┘
                           ▼
                    transcript.txt
                 (+ transcript.json)
```

### Почему два образа

NeMo (диаризация) и GigaAM часто тянут **разные/конфликтующие** стеки PyTorch и ASR-зависимостей. Разделение:

- проще воспроизводить и обновлять каждую модель;
- обмен только файлами: WAV + JSON;
- на 16 GB VRAM гоняем **последовательно** (сначала diar, потом asr) — без борьбы за память.

### Пайплайн (стратегия C)

```text
data/monoN.wav
      │
      ▼
 preprocess (при необходимости resample → 16 kHz mono; для текущих файлов — no-op)
      │
      ▼
 Sortformer diarize → сырые сегменты (t0, t1, spk)
      │
      ▼
 postprocess: merge gap, min duration, оставить 2 спикера
      │
      ▼
 нарезка WAV по укрупнённым сегментам
      │
      ▼
 GigaAM transcribe каждый кусок
      │
      ▼
 transcript.txt  (+ json)
```

### Формат TXT

```text
[00:12] Спикер 1: добрый день
[00:18] Спикер 2: здравствуйте
```

Таймкод — **начало реплики** (`MM:SS` или `HH:MM:SS` если файл длиннее часа).

---

## Ограничения моделей

### Sortformer

- Вход: mono, 16 kHz (наши `data/mono*.wav` уже подходят)
- Выход: `begin_seconds, end_seconds, speaker_index` (или T×4 probs, кадр 80 мс)
- Модель до **4** спикеров; у нас постпроцесс ориентирован на **2**
- Обучена в основном на английском → на русском проверить на глаз на `mono1…3`

### GigaAM `v3_e2e_rnnt`

- ASR по нарезанным сегментам после merge
- Слишком короткие куски (&lt; ~0.3–0.5 с) — пропускать или приклеивать к соседу того же спикера

---

## Этапы реализации

### 0. Окружение

- [ ] Docker + NVIDIA Container Toolkit
- [ ] HF token для скачивания Sortformer, лицензия CC-BY-4.0
- [ ] Кэш весов: `data/models/diar/`, `data/models/gigaam/` (в `.gitignore`)
- [ ] `docker-compose.yml`: сервисы `diar`, `asr` (GPU), volume на `data/` и `out/`

### 1. Образ `diar`

- [ ] NeMo ASR toolkit + Sortformer
- [ ] CLI/entrypoint: вход WAV → stdout/`diar.json` сегментов
- [ ] Streaming-параметры: very high latency (chunk 340 / right 40 / fifo 40 / update 300) как старт
- [ ] Smoke на `data/mono3.wav` (самый короткий)

### 2. Образ `asr`

- [ ] GigaAM `v3_e2e_rnnt` на CUDA
- [ ] CLI/entrypoint: список кусков WAV или один файл + манифест сегментов → тексты
- [ ] Smoke: короткий нарезанный фрагмент

### 3. Оркестратор CLI (host Python, тонкий)

```bash
python -m app.recognize --audio data/mono3.wav --out out/mono3/
```

Шаги:

1. (опц.) препроцессинг
2. `docker compose run diar …` → `diar.json`
3. merge / filter → `segments.json`
4. нарезка WAV
5. `docker compose run asr …` → тексты
6. сборка `transcript.txt` (+ `transcript.json`)

### 4. Прогон и тюнинг «на глаз»

- [ ] `mono3` → `mono2` → `mono1`
- [ ] Подкрутить `merge_gap`, `min_segment`, diar latency-конфиг
- [ ] Проверить, что не плодятся «Спикер 3/4»; при появлении — слить/отсечь

### 5. Вне скоупа (пока)

- Live / WebSocket
- API / UI
- Именование спикеров (не `Спикер 1/2`)
- 3+ спикеров
- Стерео «канал = спикер»
- Автоматические метрики (SER/DER/WER)
- Fine-tune моделей

---

## Структура репозитория (черновик)

```text
gigaam-recogn2/
  PLAN.md
  README.md
  docker-compose.yml
  diar/
    Dockerfile
    requirements.txt
    app/…                 # Sortformer inference
  asr/
    Dockerfile
    requirements.txt
    app/…                 # GigaAM inference
  app/                    # host orchestrator
    preprocess.py
    align.py              # merge сегментов, overlap → dominant
    recognize.py          # CLI
    export_txt.py
  data/
    mono1.wav
    mono2.wav
    mono3.wav
    models/               # gitignore, кэш весов
  out/                    # gitignore, результаты
```

Пример артефактов прогона:

```text
out/mono3/
  diar.raw.json
  segments.json
  transcript.json
  transcript.txt
```

---

## Риски

| Риск | Митигация |
|------|-----------|
| Sortformer слабее на русском | Smoke на `mono3`, тюнинг конфига; смена diar-модели только если глаз скажет «плохо» |
| Конфликт зависимостей | Уже заложено: **два Docker-образа** |
| Много коротких сегментов → плохой ASR | Merge (стратегия C), min duration |
| VRAM 16 GB на обе модели | Последовательный запуск контейнеров |
| Лишние спикеры 3/4 | Постпроцесс: оставить top-2 по суммарной длительности / слить редкие |

---

## Критерий готовности MVP

- [ ] `python -m app.recognize --audio data/mono3.wav --out out/mono3/` → `transcript.txt`
- [ ] На всех трёх `data/mono*.wav` результат читаем **на глаз**: смена спикеров выглядит разумно, текст узнаваем
- [ ] Документированы `docker compose build` и команда прогона в `README.md`

---

## Порядок работ

1. ~~Зафиксировать решения~~ (этот документ)
2. Dockerfile `diar` + smoke на `mono3`
3. Dockerfile `asr` + smoke
4. Host CLI оркестратор (стратегия C, TXT с таймкодами)
5. Прогон `mono3` → `mono2` → `mono1`, правка порогов
6. README


---

## Статус реализации

- [x] Docker-образы `diar` (NeMo+Sortformer) и `asr` (GigaAM), torch **cu128** / RTX 5060 Ti
- [x] Host CLI `recognize.py` (стратегия C, TXT с таймкодами)
- [x] Smoke на `data/mono3.wav` → `out/mono3/transcript.txt` (2 спикера, читаемый диалог)
- [ ] Прогон `mono2` / `mono1` и тюнинг порогов на глаз
