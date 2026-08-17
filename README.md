# GigaAM + Sortformer (offline, телефония)

Распознавание моно-записи разговора: **Sortformer** (диаризация) + **GigaAM v3_e2e_rnnt** (ASR).  
Для переведённых звонков (3+ участника) включён **transfer-split** (по умолчанию).

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

# Опционально: второй проход Qwen3-ASR (соседний ../qwen3-asr-campplus)
# → transcript_qwen.txt + dual_for_llm.md (два диалога рядом для LLM)
python3 recognize.py --audio data/mono3.wav --out out/mono3 --dual-qwen
```

Результат: `out/mono3/transcript.txt`

```text
[00:12] Спикер 1: ...
[00:18] Спикер 2: ...
```

Также: `diar.raw.json`, `segments.json`, `transcript.json`.  
При срабатывании перевода: `transfer_split.json`, `diar.full.json`, `transfer_parts/`.  
С `--dual-qwen`: `transcript_qwen.txt`, `asr_qwen.json`, `dual_for_llm.md`.

## Регрессионный набор

Пять фиксированных звонков + эталонные `transcript.txt` — прогон после изменений:

```bash
python3 tests/regression/run.py
# обновить эталоны после осознанного улучшения:
python3 tests/regression/run.py --update-golden
```

Подробности: [`tests/regression/README.md`](tests/regression/README.md).

## Архитектура

| Сервис | Образ | Роль |
|--------|-------|------|
| `diar` | NeMo + Sortformer | сегменты спикеров |
| `asr` | GigaAM `v3_e2e_rnnt` | текст по сегментам |
| host CLI | `recognize.py` | merge, нарезка, TXT, transfer-split |

Контейнеры запускаются **последовательно** (сначала diar, потом asr; при переводе — ещё diar head/tail и второй ASR).

## Определение перевода (transfer-split)

По умолчанию `TRANSFER_SPLIT=1`. Пайплайн: полный diar → черновик ASR → **поиск точки разреза** → при находке re-diar head/tail → склейка спикеров → финальный ASR.

### Что используем сейчас

| Сигнал | Откуда | Роль |
|--------|--------|------|
| **Cue (слова)** | черновик `transcript` | обязательный триггер по умолчанию |
| **Audio hold** | energy VAD по `audio_16k.wav` | тишина / периодические гудки hold — *где* резать после cue |
| **Пауза diar** | дыры между speech-сегментами Sortformer | запасной якорь, если audio hold нет |
| **Cue-only** | конец cue + ~0.4 с | холодный перевод почти без паузы |

**Cue** (regex, регистр не важен), default:

- `перевед`, `переключ`, `соедин` (соединить / соединяю / …)
- `передам вас|ваш` (с `\b`, чтобы не ловить «передам вашему…»), `оставайтесь на линии`

Порядок выбора точки разреза (если есть хотя бы один cue):

1. **`audio_silence_after_cue` / `audio_music_after_cue`** — hold по wav (≥ `TRANSFER_AUDIO_HOLD_SEC`, default 5 с) в окне после cue.
2. **`last_gap_after_cue`** — последняя длинная diar-пауза (≥ `TRANSFER_GAP_SEC`, default 8 с) в окне `TRANSFER_HOLD_WINDOW_SEC` (60 с).
3. **`soft_gap_after_cue`** — первая diar-пауза ≥ 1.5 с в том же окне.
4. **`cue_only`** — разрез сразу после последней cue-фразы (нет заметной тишины).

Audio hold (`app/hold_detect.py`): глубокая тишина (низкий перцентиль энергии) + периодические сильные гудки (ringback/hold, период ~1.5–5.5 с). Артефакт: `hold_detect.json`.

Склейка half-файлов: `TRANSFER_CONTINUITY=first_new` — первый голос после разреза = новый агент; клиент сопоставляется с самым «длинным» спикером head.

Артефакты: `transfer_split.json` (`reason`, `split_t`, `cue_hits`), `hold_detect.json`, исходный diar → `diar.full.json`.

### Что сознательно не делаем (пока)

| Идея | Статус |
|------|--------|
| Резать только по самой длинной mid-call паузе **без cue** | Выкл. (`TRANSFER_GAP_ONLY=0`) — много ложных срабатываний |
| Резать только по audio hold **без cue** | Выкл. (`TRANSFER_HOLD_ONLY=0`) |
| Спектральный детект произвольной hold-музыки (не гудки) | Не реализовано; гудки/тишина — да |

### CLI / env

```bash
# выключить
python3 recognize.py --audio … --out … --no-transfer-split
# или TRANSFER_SPLIT=0
```

| Переменная | Default | Смысл |
|------------|---------|--------|
| `TRANSFER_SPLIT` | `1` | вкл/выкл |
| `TRANSFER_GAP_SEC` | `8` | длинный diar-hold (сек) |
| `TRANSFER_AUDIO_HOLD` | `1` | energy VAD / гудки как третий сигнал |
| `TRANSFER_AUDIO_HOLD_SEC` | `5` | мин. длительность audio hold |
| `TRANSFER_HOLD_WINDOW_SEC` | `60` | окно поиска паузы/hold после cue |
| `TRANSFER_MARGIN_SEC` | `15` | не резать у краёв файла |
| `TRANSFER_CONTINUITY` | `first_new` | склейка head↔tail (`longest` — устаревший вариант) |
| `TRANSFER_MIN_SPK_SEC` / `SHARE` | `4` / `0.03` | мягче фильтр спикеров на втором ASR |
| `TRANSFER_GAP_ONLY` | `0` | разрешить split без cue по longest diar gap |
| `TRANSFER_HOLD_ONLY` | `0` | разрешить split без cue по longest audio hold |
| `TRANSFER_CUES` | см. выше | `\|`-список regex |

Код: `app/transfer_split.py`, `app/hold_detect.py`, вызов из `app/recognize.py`.

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
- `MAX_SPEAKERS` — потолок спикеров после diar (default `4`; Sortformer до 4)
- `MIN_SPK_SEC` / `MIN_SPK_SHARE` — 3-й/4-й спикер только если речь ≥ N сек и доля ≥ share (default `10` / `0.06`). Топ-2 всегда остаются.
- `TRANSFER_SPLIT` и родственные — см. [Определение перевода](#определение-перевода-transfer-split)
- `MAX_ASR_SEC` — нарезка длинных реплик по паузам diar (default `20`)
- `DIAR_MODEL` — по умолчанию `nvidia/diar_streaming_sortformer_4spk-v2.1`
- `DIAR_CHUNK_LEN` и др. — streaming-конфиг Sortformer (very high latency)

## Кэш моделей

- GigaAM: `data/models/gigaam/`
- HuggingFace / NeMo: `data/models/hf`, `data/models/nemo`
