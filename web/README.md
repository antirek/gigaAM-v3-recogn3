# Web UI (Vue3 + Express + MongoDB + Caddy)

Просмотр результатов распознавания и саммари звонков.

## Стек

| Слой | Технологии |
|------|------------|
| API | Node.js, Express, Mongoose |
| БД | MongoDB |
| UI | Vue 3 (Composition API), Vite |
| Reverse proxy | **Caddy** (статика + `/api` → `web-api`) |

## Docker (рекомендуется)

Из корня репозитория:

```bash
# Mongo + API + Caddy (UI на :8080)
docker-compose up -d mongo web-api web

# Импорт батча (после recognize + summarize)
docker-compose run --rm web-api node scripts/import-batch.js /out/outgoing_answered_gt30_2026-08-19
```

Открыть: **http://localhost:8080**

> В Docker `mongo` не публикует порт на хост (чтобы не конфликтовать с другими Mongo). API ходит по сети compose: `mongodb://mongo:27017/...`.

Переменные (`.env`):

| Переменная | Default | Смысл |
|------------|---------|--------|
| `WEB_PORT` | `8080` | порт Caddy (UI + прокси `/api`) |
| `WEB_CORS_ORIGIN` | `http://localhost:8080` | CORS для прямых запросов к API |

Архитектура:

```text
браузер → web (Caddy :80)
            ├─ /        → Vue static (dist)
            └─ /api/*   → web-api:3000 → MongoDB
```

В `web/server` и `web/client` лежит `.npmrc` с `registry=https://registry.npmjs.org/` — Docker-сборка не ходит в приватный Mobilon registry.
## Локальная разработка (без Docker)

```bash
docker-compose up -d mongo

cd web/server && npm install && npm run dev
cd web/client && npm install && npm run dev
```

UI: http://localhost:5173 (Vite proxy `/api` → `:3000`)

Импорт:

```bash
cd web/server
npm run import -- out/outgoing_answered_gt30_2026-08-19
```

## API

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/api/health` | healthcheck |
| GET | `/api/calls` | список звонков (фильтры: `batchTag`, `date`, `escalation`, `severity`, `reason`, `q`, `page`, `limit`) |
| GET | `/api/calls/:callId` | звонок с transcript + summary |
| GET | `/api/batches` | список импортированных батчей |
| GET | `/api/batches/:batchTag` | batch_summary за день |
| POST | `/api/import/batch` | загрузка JSON `{ batchTag, calls[], batchSummary }` |
| POST | `/api/import/batch-from-path` | импорт с диска `{ path }` (в контейнере: `/out/...`) |

## Тесты API

```bash
cd web/server
docker-compose up -d mongo
npm test
```
