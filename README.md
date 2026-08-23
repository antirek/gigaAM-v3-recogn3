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
```

Результат: `out/mono3/transcript.txt`

```text
[00:12] Спикер 1: ...
[00:18] Спикер 2: ...
```

Также: `diar.raw.json`, `segments.json`, `transcript.json`.  
При срабатывании перевода: `transfer_split.json`, `diar.full.json`, `transfer_parts/`.

## Регрессионный набор

Пять фиксированных звонков + эталонные `transcript.txt` — прогон после изменений:

```bash
python3 tests/regression/run.py
# обновить эталоны после осознанного улучшения:
python3 tests/regression/run.py --update-golden
```

Подробности: [`tests/regression/README.md`](tests/regression/README.md).

## Web UI (просмотр звонков)

После `recognize` + `summarize-call` + `summarize-batch` данные можно загрузить в **MongoDB** и смотреть в браузере:

- таб **Список звонков** — фильтры, эскалации, модалки (диалог / JSON / саммари);
- таб **Саммари за день** — `batch_summary` + список эскалаций.

Стек: Express + Mongoose + Vue 3, в Docker — **Caddy** (статика + прокси `/api`). Подробности: [`web/README.md`](web/README.md).

```bash
docker-compose up -d mongo web-api web
# → http://localhost:8080

docker-compose run --rm web-api node scripts/import-batch.js /out/outgoing_answered_gt30_2026-08-19
```

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
- LLM GGUF: `data/models/llamacpp/`

## Используемые модели

Все три рабочие модели в проде крутятся на **GPU (CUDA / VRAM)**, не на CPU:  
`diar` и `asr` — `runtime: nvidia`, у ASR явно `DEVICE=cuda`; LLM — `llama.cpp` с full GPU offload (`-ngl 99`).  
CPU возможен только как fallback / ручной `DEVICE=cpu` (для телефонии не используем).

Размеры на диске — по файлам в `data/models/` на этой машине.  
**VRAM** — ориентир для нашего режима (одна модель на GPU, без параллельного diar+ASR+LLM).

| Роль | Модель | Публикация | Диск | Устройство | VRAM (ориентир) | Где лежит |
|------|--------|------------|------|------------|-----------------|-----------|
| **ASR** | [GigaAM](https://huggingface.co/ai-sage/GigaAM-v3) `v3_e2e_rnnt` (~220–240M) | **2025-11** (GigaAM-v3 / e2e) | **0.45 GB** (`v3_e2e_rnnt.ckpt` + tokenizer) | **GPU** | **~1–2 GB** (fp16, короткие чанки) | `data/models/gigaam/` |
| **Diarization** | [nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) (117M) | **2025-10-22** (HF `createdAt`) | **0.47 GB** (`.nemo`) | **GPU** | **~0.5–1.5 GB** | `data/models/hf/hub/models--nvidia--…` |
| **LLM** | [GigaChat3.1-10B-A1.8B](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF) **q6_K** (MoE 10B / ~1.8B active) | **2026-03-21** (GGUF repo) | **8.78 GB** (`.gguf`) | **GPU** | **~10–12 GB** (full offload, `-c 16384`) | `data/models/llamacpp/GigaChat3.1-10B-A1.8B-q6_K.gguf` |

**Суммарно на диске (основные веса):** ≈ **9.7 GB**.

Заметки:
- diar и ASR на одной карте **по очереди** (compose запускает контейнеры последовательно); LLM (`llamacpp`) держит GGUF в VRAM постоянно — не гонять recognize параллельно с summarize.
- опциональный `extract-hybrid` тянет GLiNER (`fulstock/gliner-nerel-finetuned`) отдельно и в основной дневной путь **не входит**.
- железная цель пайплайна: **RTX 5060 Ti 16 GB**.

## Пост-обработка текста (LLM/LLM-free извлечение фактов)

После `recognize.py` у нас уже есть `out/<id>/transcript.txt` в формате:

```text
[MM:SS] Спикер N: ...
```

Дальше работает отдельный блок в `./llm/` (контейнер сервиса `llm` + `llamacpp` в compose).

### Сводка: все постобработки и чем выполняем

| # | Шаг (CLI) | Что делает | Чем выполняем | Выход |
|---|-----------|------------|---------------|--------|
| 1 | `refine` | Нормализация транскрипта (пунктуация, пробелы, мелкие ASR-правки) | **Rules** (`llm_rules`); опционально LLM-патчи через **llama.cpp / GigaChat3.1-q6_K** (`safe`/`smart`) + снова rules | `transcript` refined + debug edits |
| 2 | `extract` | Факты: телефоны, адреса, суммы, commitments | **GigaChat3.1** (llama.cpp, JSON schema) + Python **sanitize/grounding** телефонов | `phones/addresses/amounts/commitments` |
| 3 | `extract-natasha` | Те же факты без LLM (без commitments) | **Natasha** (детерминированно) + общий phone grounding | `phones/addresses/amounts` |
| 4 | `roles` | Роли спикеров `ivr` / `client` / `agent` | **GigaChat3.1** (llama.cpp) + whitelist sanitize | `speakers[]` + roles |
| 5 | `extract-hybrid` | Факты + люди/орги/машины/мессенджеры | **Natasha** + **GLiNER1** (`fulstock/gliner-nerel-finetuned`); commitments — не здесь | hybrid JSON |
| 6 | `summarize-call` | Резюме одного звонка + эскалация супервайзеру | **GigaChat3.1**: (a) summary JSON, (b) отдельный escalation-pass; **IVR/hold pre-filter** + keyword boost; fallback `rules` если `LLM_FALLBACK_TO_RULES=1` | `call_summary.json` / `.md` |
| 6a | `summarize-call --escalation-only` | Только пересчёт `escalation` | тот же escalation-стек, существующий summary не переписывает narrative | обновляет `escalation` в `call_summary.*` |
| 7 | `summarize-batch` | Дневной отчёт по папке звонков | **GigaChat3.1** map-reduce (чанки → reduce); список **`supervisor_escalations`** собирается **детерминированно** из per-call JSON | `batch_summary.json` / `.md` |

**Рантайм LLM:** сервис `llamacpp` (`ghcr.io/ggml-org/llama.cpp:server-cuda13`) + GGUF `GigaChat3.1-10B-A1.8B-q6_K.gguf`, alias `GigaChat3.1-10B-A1.8B-q6_k`. Клиент: контейнер `llm`, `LLM_BACKEND=llamacpp`.

**Не LLM / не в таблице выше (но рядом по пайплайну):**
- сам `recognize.py` (Sortformer + GigaAM, transfer-split, hold-detect) — см. [Архитектура](#архитектура).

Типичный дневной путь для аналитики звонков: **recognize → summarize-call → summarize-batch**.  
Extract/roles/hybrid — по необходимости (факты, роли, NER), не обязательны для batch-отчёта.

### Какие модели сейчас используются

#### Llama.cpp runtime (опционально, для ролей/обязательств)

`docker-compose.yml` на сервисе `llamacpp` стартует:

- GGUF: `data/models/llamacpp/GigaChat3.1-10B-A1.8B-q6_K.gguf`
- Served name/alias: `GigaChat3.1-10B-A1.8B-q6_k`

`llm` (клиент) использует:
- `LLM_BACKEND=llamacpp`
- `LLAMACPP_MODEL=GigaChat3.1-10B-A1.8B-q6_k`

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

6) `summarize-call` / `summarize-batch` (GigaChat3.1 через llama.cpp)

Дальше (после `recognize.py`) используем **GigaChat3.1-10B-A1.8B** (MoE 10B / 1.8B active, GGUF **q6_K**) через llama.cpp для резюме звонков и эскалаций. На RTX 5060 Ti 16 GB (`-c 16384`, full offload) укладывается в VRAM с запасом.

#### `summarize-call`

Команда в `llm/llm_cli.py` (в контейнере — entrypoint `llm`):
```bash
docker-compose run --rm --no-deps -e LLM_FALLBACK_TO_RULES=0 llm summarize-call \
  --input /out/<tag>/<id>/transcript.txt \
  --out-dir /out/<tag>/<id> </dev/null
```

Только пересчёт эскалации (intent/issues не трогает):
```bash
docker-compose run --rm --no-deps -e LLM_FALLBACK_TO_RULES=0 llm summarize-call \
  --escalation-only \
  --input /out/<tag>/<id>/transcript.txt \
  --out-dir /out/<tag>/<id> </dev/null
```

Результат:
- `call_summary.json`
- `call_summary.md`

Ключевые свойства:
- `intent` — **нарратив на русском (2–3 предложения)**, не короткий label вроде `technical/telephony`.
- `issues_detected` / `actions` — с цитатами и ролями (`agent`/`client`), где модель их даёт.
- `call_id` принудительно берётся из имени `--out-dir` (модель иногда засоряла поле).
- Рендер `.md` не печатает пустые `: ...` для `evidence`/`who`.
- `escalation` — отдельный LLM-pass после summary (см. ниже); в `.md` секция **Escalation (supervisor)**.

#### Эскалация руководителю (`escalation`)

Поле в `call_summary.json`:
```json
"escalation": {
  "needed": true,
  "severity": "high|medium|low",
  "reasons": ["complaint_threat", "billing_dispute", "agent_quality", "unresolved_repeat", "process_failure"],
  "evidence": ["короткие цитаты"],
  "summary_for_manager": "1–2 предложения для супервайзера"
}
```

Логика (не путать с L2 tech-ticket):
1. отдельный chat с узкой схемой `EscalationDecision`;
2. **IVR/hold pre-filter** — очередь («операторы заняты», «оставайтесь на линии», no-answer) → сразу `needed=false` без LLM;
3. жёсткие negatives в промпте (ожидание в IVR ≠ `process_failure`);
4. demote queue false-positives после ответа модели;
5. **keyword boost** для редких under-fire сигналов (`претензи`, `не надо оплачивать`, «две недели» + «старая линия»).

#### Защита от пустых транскриптов

Если `transcript.txt` пустой или слишком короткий, `summarize-call` **не вызывает LLM** и пишет пустую структуру с:
- `quality_notes.asr_uncertainty = "empty_transcript_skipped"`
- `escalation.needed = false`

Порог: `SUMMARIZE_MIN_TEXT_CHARS` (default `30`).

#### `summarize-batch` (дневной отчёт)

```bash
docker-compose run --rm --no-deps \
  -e LLM_FALLBACK_TO_RULES=0 \
  -e LLAMACPP_MAX_TOKENS=6000 \
  -e BATCH_SUMMARY_CHUNK_SIZE=12 \
  llm summarize-batch \
  --input-dir /out/<tag> \
  --out-dir /out/<tag> \
  --date YYYY-MM-DD </dev/null
```

Ожидается, что в `--input-dir/*/call_summary.json` уже лежат per-call summary (папки `_…` пропускаются).

Результат:
- `batch_summary.json` / `batch_summary.md`
- narrative: `executive_summary`, `key_moments`, `recurring_problems`, `positive_moments`, `potential_risks`, `top_topics`, `recommendations`
- **детерминированно** (не от LLM-выдумки): `supervisor_escalations[]`, `n_escalations` — список звонков с `escalation.needed=true`, в `.md` секция **«Эскалации руководителю»** сразу после общей картины

**Map-reduce на больших днях:** все summary разом (~десятки тысяч токенов) в контекст LLM (`-c 16384`) не влезают. Поэтому:
1. чанки по `BATCH_SUMMARY_CHUNK_SIZE` (default `12`) → дайджесты;
2. финальная склейка в дневной отчёт (`mode=map_reduce`).

Маленький день (≤ chunk size) идёт одним проходом (`single_pass`).  
Для reduce удобно `-e LLAMACPP_MAX_TOKENS=6000` (GigaChat многословный — иначе JSON может обрезаться).

#### Важные operational notes

1) **`LLM_FALLBACK_TO_RULES=1` (default в compose)**  
Если llama.cpp недоступен/упал, `summarize-call` молча отдаёт rules-backend: ярлыки `technical/telephony`, `service_request` и шаблонные issues. На большом батче это выглядело как «больше половины плохих summary».  
Для продакшен-прогонов лучше `-e LLM_FALLBACK_TO_RULES=0`, чтобы сбой был виден.

2) **GPU contention**  
`llamacpp` держит GGUF в VRAM постоянно. Параллельный `recognize` (diar+ASR) на той же карте может ронять LLM-вызовы → снова rules-fallback. Надёжнее: сначала весь recognize, потом отдельным проходом summarize (или не держать diar/asr одновременно с LLM).

3) **bash + `docker-compose run` в цикле**  
`docker-compose` читает stdin и может «съесть» список файлов в `while read`. В циклах: `docker-compose run ... </dev/null`.

4) **Смена модели**  
После замены GGUF / alias в `docker-compose.yml` — `docker-compose up -d --force-recreate llamacpp` и при правках кода `llm/` — `docker-compose build llm` (исходники в образ не монтируются).

#### Пример дневного пайплайна (API Mobilon)

Фильтр журнала: `direction=external|outgoing`, `status=ANSWERED`, `duration > 30`, запись по `record_url`.

```text
data/calls/<tag>/*.mp3
  → python3 recognize.py --audio … --out out/<tag>/<stem>
  → llm summarize-call (на каждый transcript; внутри + escalation)
  → llm summarize-batch
```

Прогон **2026-08-19** (исходящие ANSWERED >30с):
- **96** звонков, аудио **12 961 с ≈ 3 ч 36 мин**
- полный `recognize + summarize`: **~1 ч 13 мин** wall-clock (до миграции на GigaChat)
- 2 пустых transcript → skip LLM
- миграция на **GigaChat3.1-q6_K**: полный пересчёт 96 `summarize-call` **~16.5 мин**; `summarize-batch` map-reduce (8 чанков) **~1–2 мин**
- после калибровки эскалации (IVR pre-filter + промпт): **`n_escalations ≈ 32`** из 94 usable (без hold/IVR false positives); жёсткие кейсы (претензия, billing refuse, process «старая линия») сохраняются
- пересчёт только эскалации: `--escalation-only` на 96 **~7 мин**

Артефакты:
- `data/calls/outgoing_answered_gt30_2026-08-19/`
- `out/outgoing_answered_gt30_2026-08-19/<stem>/transcript.txt` + `call_summary.*`
- `out/outgoing_answered_gt30_2026-08-19/batch_summary.md`

Ограничение качества:
- narrative «о чём день» обычно ок;
- **агрегатные цифры в `executive_summary`** LLM может округлять/путаться — для ops смотреть детерминированный блок `supervisor_escalations` / `n_escalations`;
- `agent_quality` / `unresolved_repeat` всё ещё могут быть шумнее идеальной QoS-очереди (~10–15) — при необходимости ужесточать промпт и перегонять `--escalation-only`.

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
