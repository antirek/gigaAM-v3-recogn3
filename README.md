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

## Пост-обработка текста (LLM/LLM-free извлечение фактов)

После `recognize.py` у нас уже есть `out/<id>/transcript.txt` в формате:

```text
[MM:SS] Спикер N: ...
```

Дальше работает отдельный блок в `./llm/` (контейнер сервиса `llm` + `llamacpp` в compose).

### Какие модели сейчас используются

#### Llama.cpp runtime (опционально, для ролей/обязательств)

`docker-compose.yml` на сервисе `llamacpp` стартует:

- GGUF: `data/models/llamacpp/T-lite-it-2.1-Q8_0.gguf`
- Served name/alias: `T-lite-it-2.1-q8_0`

`llm` (клиент) использует:
- `LLM_BACKEND=llamacpp`
- `LLAMACPP_MODEL=T-lite-it-2.1-q8_0`

Почему llama.cpp:
- vLLM убран: на `refine`/JSON schema он давал нестабильный JSON (пустые/обрывающиеся `splits` и раздувание структуры под `max_tokens`).

#### Детерминированный слой Natasha (LLM-free, факты)

Мы добавили локальный модуль Natasha для извлечения **фактов** без LLM:
- телефоны (через сбор диктовки + валидацию формата)
- адреса
- суммы денег

LLM при этом остаётся для:
- ролей спикеров (`ivr/client/agent`)
- обязательств/обещаний (commitments)

### CLI в `llm/`

Главный вход — `llm/llm_cli.py` (в контейнере это `python3 /work/llm/llm_cli.py`).

1) `refine`
- Упрощает/нормализует `transcript.txt` детерминированными правилами (пунктуация/пробелы/мелкие правки).
- Команда: `python3 llm/llm_cli.py refine --input ... --output ... --debug-output ...`

2) `extract` (LLM, full schema)
- Извлекает `phones`, `addresses`, `amounts`, `commitments`.
- Важно: после ответа LLM в Python есть строгая `sanitize`: телефоны фильтруются regex + “grounding” по цифрам в исходном транскрипте.
- Команда:
  - `python3 llm/llm_cli.py extract --input ... --output ... --call-id ...`

3) `extract-natasha` (LLM-free facts)
- Вытаскивает только `phones`, `addresses`, `amounts`.
- `commitments` не заполняются (это остаётся под LLM).
- Команда:
  - `python3 llm/llm_cli.py extract-natasha --input ... --output ... --call-id ...`

4) `roles` (LLM)
- LLM-этикетка ролей по репликам: `ivr | client | agent | unknown`.
- Результат потом санитарится: “лишние” спикеры выкидываются, роли ограничены whitelist.
- Команда:
  - `python3 llm/llm_cli.py roles --input ... --output ... --call-id ...`

5) `extract-hybrid` (Natasha facts + GLiNER1 people/org)
- Использует Natasha как основу (`phones/addresses/amounts`)
- Потом GLiNER (версия 1, NEREL fine-tune) добавляет:
  - `people`
  - `organizations`
  - `cars`
  - `messengers`
- Команда:
  - `python3 llm/llm_cli.py extract-hybrid --input ... --output ... --call-id ...`

6) `summarize-call` / `summarize-batch` (T-lite через llama.cpp)

Дальше (после `recognize.py`) используем модель **T-lite-it-2.1-q8_0** (llama.cpp runtime) для построения резюме звонков.

#### `summarize-call`

Команда в `llm/llm_cli.py`:
- `python3 llm/llm_cli.py summarize-call --input out/<id>/transcript.txt --out-dir out/<tag>/<id>/`

Результат:
- `call_summary.json`
- `call_summary.md`

Ключевые улучшения, которые мы сделали:
- `intent` теперь обязан быть **нарративным** (2–3 предложения на русском), а не коротким label.
- `issues_detected` и `actions` стали “плотнее”: timeline расширяется, а `issues_detected[*].evidence` и `actions[*].who`/`deadline` запрашиваются/валидируются, чтобы в .md не было пустых `: ...`.
- исправили редкий баг с `call_id`: модель иногда “засоряла” поле лишним текстом, поэтому CLI принудительно доверяет `call_id` из имени директории.
- улучшили рендер `call_summary.md`, чтобы пустое `evidence`/`who` не выглядело как “`- : ...`”.

#### Защита от пустых транскриптов

Если `transcript.txt` пустой или слишком короткий, `summarize-call` **не вызывает LLM** и сразу пишет пустую структуру:
- `intent: ""`
- пустые `topics/timeline/entities/actions/issues_detected`
- `quality_notes.asr_uncertainty = "empty_transcript_skipped"`

Порог задаётся переменной:
- `SUMMARIZE_MIN_TEXT_CHARS` (по умолчанию `30`)

#### `summarize-batch`

Команда:
- `python3 llm/llm_cli.py summarize-batch --input-dir out/<tag>/ --out-dir out/<tag>/ --date YYYY-MM-DD`

Ожидается, что внутри `--input-dir` для каждого звонка лежит папка с `call_summary.json`.

Результат:
- `batch_summary.json`
- `batch_summary.md` (единый “дневной” отчёт)

### Как устроена “защита от галлюцинаций” для телефонов

И в LLM-`extract`, и в Natasha используется общий принцип:

1) Формат RU телефона:
- допускаются только 11 цифр (начинается на `7/8`) или 10 цифр (начинается на `9`)
- запрет на “слишком нулевую”/сомнительную вариацию (`0{5,}` и т.п.)
- минимальная уникальность цифр в теле

2) Grounding телефонов:
- временные метки `[MM:SS]` вычищаются
- берутся токены с цифрами из транскрипта
- телефон принимается только если он “объясним” набором цифр в тексте (coverage по токенам >= порога)

Телефоны, которые не проходят — попадают в `notes` как `dropped_phones:...`.

### Что получилось на регрессионном наборе (5 звонков)

Регрессия: `tests/regression/golden/*` (5 фиксированных звонков).

#### LLM `extract` (llama.cpp) — базовая точность по фактам

На этих 5 звонках LLM-`extract` (с Python grounding/sanitize):
- `85013602`: телефон **89520647701**, commitments “Я передам информацию, с вами свяжутся.”
- `eda54d05`: адрес **город Тверь, улица Шишкова, дом 104**, commitments “Я буду ждать”
  - “похожий” телефон LLM иногда придумывал, но grounding отфильтровал его (см. `notes: dropped_phones:...`)
- `6b9320d3`: сумма **13600 RUB**
- `a7144842`: commitments “приезжайте” (адреса по версии LLM были слабее/похожими)
- `ee1e3c47`: слабый адрес “корпус 1”, commitments “будем вас ждать”

#### Natasha `extract-natasha` — “лучше по фактам, чем LLM”

На тех же 5 звонках Natasha стабильно (без галлюцинаций) дала:
- `85013602`: телефон **89520647701**
- `eda54d05`: адрес **город Тверь, улица Шишкова, дом 104**
- `6b9320d3`: сумма **13600 RUB**
- `a7144842` / `ee1e3c47`: пока пусто (в основном из-за слабости адресных сигналов в ASR и того, что Natasha не пытается угадывать “под адрес”)

При этом Natasha оставляет `commitments` пустыми — это сознательно: commitments остаются под LLM.

### Эксперименты с “слоем сущностей” (NER/IE): что пробовали и почему не взяли в прод

Цель экспериментов была: улучшить извлечение людей/организаций/адресов/денег/обязательств без LLM-галлюцинаций.

#### DeepPavlov `ner_rus_bert` / `ner_rus_convers_distilrubert`

Оказались неадекватными как замена Natasha/LLM на наших разговорных звонках:
- в основном ловили имена (PER), но адреса/“обязательства” не структурировали
- остальной доменный шум (ASR/эхо) давал ложные сущности

Поэтому не интегрировали.

#### Pullenti

Pullenti хорошо структурировал некоторые адресные/денежные штуки, но:
- имена/обязательства на наших ASR-разговорах часто отсутствовали или “склеивались” не так
- телефоны/номерные кодировки — отдельная боль (ASR диктовки vs true digits)

Остался как эксперимент/возможный будущий “rules+IE” модуль, но не подменяет текущий стек.

#### GLiNER1 (NEREL fine-tune)

GLiNER (версия 1) на `fulstock/gliner-nerel-finetuned` работал заметно лучше как слой:
- имена людей (PER) — извлекаются
- организации — извлекаются
- “салон/отдел” как ORG/PRODUCT — тоже извлекается
- модель авто/номенклатура — извлекается как `PRODUCT`

Минусы:
- телефонные диктовки попадают как `NUMBER`, а не как phone
- обязательства/commitments как структурированная сущность не были выделены стабильно без тонкой настройки

Поэтому GLiNER1 сейчас используется только в гибридном извлечении как “добавка людей/org/машин” (`extract-hybrid`), а commitments остаются под LLM.

#### GLiNER2 (multi-task) — НЕ включили

Мы пробовали GLiNER2 (base `fastino/gliner2-base-v1`) в двух A/B вариантах на наших 5 звонках.

Результат:
- A/B v1 (`entity_types=['promise','callback','person']` + cue-фильтр): `pred=None` на 4 звонках из 4 с gold обязательствами
- A/B v2 (русские entity_types + threshold ниже): улучшение только на одном звонке, где нашли **часть** обязательства:
  - `85013602`: найдено “с вами свяжутся” вместо полного “Я передам информацию, с вами свяжутся.”
  - остальные звонки: снова `pred=None`

Из-за низкой устойчивости в условиях разговорного ASR “commitments как сущность” из GLiNER2 не получились как надёжная замена LLM.

После эксперимента GLiNER2 пакет и артефакты A/B были удалены (оставлен только GLiNER1/Natasha/LLM стек).

### Где лежат результаты извлечений

- `out/llm_extract/<id>.json` — основной LLM extraction schema (phones/addresses/amounts/commitments)
- `out/llm_extract_natasha/<id>.json` — Natasha facts (phones/addresses/amounts)
- `out/llm_roles/<id>.json` — роли (ivr/client/agent)
- `out/llm_extract_hybrid/<id>.json` — hybrid facts + people/org/car/messengers
