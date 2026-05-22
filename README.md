# PDF → Clean PDF Converter

Self-hosted, single-user web app that converts **image-based PDF files** to clean, re-typeset PDF files using **Google Gemini** for OCR.

- ✅ OCR via Google Gemini (`gemini-2.5-flash-lite` by default)
- ✅ Languages: Traditional Chinese, Simplified Chinese, Japanese, Korean, English (and 100+ others)
- ✅ Auto-detects horizontal / vertical text layout per page
- ✅ Clean PDF with correct CJK font/CMap per detected language
- ✅ Re-embeds images, preserves hyperlinks, headings, TOC, footnotes, page numbers
- ✅ Async job queue with Start / Pause / Stop / Delete / Retry controls
- ✅ Google OAuth2 authentication (single-user, allowlist by email)
- ✅ One Gemini API call per PDF page (efficient, low cost / quota)
- ✅ Per-page OCR caching — pause/resume without spending extra quota

---

## Why Gemini?

The previous PaddleOCR / Surya implementations needed too much RAM for the Zeabur server. Gemini moves OCR off-server entirely — the worker just sends each page image to Google's API and receives structured JSON back. The Zeabur worker now uses **under 1 GB RAM** and needs no GPU or PyTorch.

Trade-off: each PDF page = 1 Gemini API call, so **daily free-tier quota matters**. The default model `gemini-2.5-flash-lite` allows **1,000 requests per day for free**. If you pause or a job fails partway through, the OCR results for completed pages are cached — resuming costs zero extra quota for those pages.

---

## Architecture

```
Browser → FastAPI (auth + UI + queue control)
                  ↕
         In-process store (fakeredis, or external Redis if REDIS_URL is set)
                  ↕
         Worker thread (Gemini API client, runs inside the same container)
                  ↕
    Google Gemini API  (https://generativelanguage.googleapis.com)
```

The API and Worker run as a **single process** — the worker is a background daemon thread started automatically when the app boots. No separate Redis service is needed; state is kept in-process via [fakeredis](https://github.com/cunla/fakeredis-py). If you set `REDIS_URL`, an external Redis is used instead (useful if you want persistent state across restarts).

---

## Prerequisites

- Zeabur server: any plan with at least **2 GB RAM** is sufficient (the $3/mo 4 GB plan works comfortably)
- A **Google account** (Workspace or personal Gmail) for OAuth2 login
- A **Gemini API key** from Google AI Studio

---

## Step 1 — Get a Gemini API Key

1. Go to [https://aistudio.google.com](https://aistudio.google.com)
2. Sign in with your Google account
3. Click **Get API key** in the left sidebar
4. Click **Create API key → Create API key in new project**
5. Copy the key — it looks like `AIzaSy...` (~39 characters)

The free tier for `gemini-2.5-flash-lite` (the default model shipped in `config.yaml`) gives you:
- **15 requests per minute**
- **1,000 requests per day**

The quota resets at midnight Pacific Time. Each PDF page = 1 request.

If you prefer higher accuracy at the cost of lower quota, switch `model_name` in `config.yaml` to `gemini-2.5-flash` (10 RPM, 250 RPD).

---

## Step 2 — Google OAuth2 Setup (for app login)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Use the same project Gemini created (or any project)
3. **APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client ID**
4. Application type: **Web application**
5. Authorised redirect URI: `https://YOUR-ZEABUR-DOMAIN/auth/callback`
6. Copy the **Client ID** and **Client Secret**

---

## Step 3 — Environment Variables (Zeabur UI)

| Variable | Value |
|---|---|
| `GEMINI_API_KEY` | API key from Step 1 |
| `GOOGLE_CLIENT_ID` | OAuth client ID from Step 2 |
| `GOOGLE_CLIENT_SECRET` | OAuth client secret from Step 2 |
| `ALLOWED_EMAIL` | Your Gmail address |
| `SECRET_KEY` | Random 32+ char string (`openssl rand -hex 32`) |
| `BASE_URL` or `APP_BASE_URL` | `https://YOUR-ZEABUR-DOMAIN` (no trailing slash) |

> **No Redis service needed.** The app uses in-process storage by default. If you want persistent state across container restarts, add `REDIS_URL` pointing to an external Redis instance.

---

## Step 4 — Deploy

```bash
git clone https://github.com/YOUR-USERNAME/pdf2epub.git
cd pdf2epub

# Local dev: create a .env file
cat > .env << EOF
GEMINI_API_KEY=AIzaSy...
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
ALLOWED_EMAIL=you@gmail.com
SECRET_KEY=$(openssl rand -hex 32)
BASE_URL=https://your-zeabur-domain.com
EOF

docker compose up -d
docker compose logs -f
```

On Zeabur, push the repo to GitHub, create a new project, connect the repo, and set the env variables in Zeabur's UI. Zeabur detects the root `Dockerfile` and deploys it as **a single service** — no Redis or Worker service needed.

---

## Step 5 — Verify

1. `https://YOUR-ZEABUR-DOMAIN/health` → `{"status":"ok","redis":true}`
2. `https://YOUR-ZEABUR-DOMAIN` → login page
3. Sign in with the allowlisted Gmail
4. Upload a PDF, click **Start**, watch progress
5. When done, click **↓ Clean PDF** to download

---

## Configuration (`config.yaml`)

```yaml
ocr:
  engine: gemini
  model_name: "gemini-2.5-flash-lite"   # default; change to gemini-2.5-flash for higher accuracy
  rpm_limit: 15                          # match your Gemini tier
  rpd_limit: 1000
  max_retries: 3
  request_timeout_s: 120
  confidence_threshold: 0.7
  dpi: 400                               # rasterization DPI

pipeline:
  max_pdf_size_mb: 100
  page_batch_size: 5
  upload_retention_hours: 24
  output_retention_days: 7
  tmp_cleanup_on_complete: true

server:
  max_concurrent_jobs: 1
  port: 8080
  job_history_limit: 10
```

### Switching to `gemini-2.5-flash` (higher accuracy, lower daily quota)

```yaml
ocr:
  model_name: gemini-2.5-flash
  rpm_limit: 10
  rpd_limit: 250
```

Then `docker compose restart app` to apply.

---

## Memory Budget

| Component | Idle | Peak |
|---|---|---|
| OS + existing services | ~1.2 GB | ~1.2 GB |
| FastAPI + in-process store | ~200 MB | ~200 MB |
| Worker thread (Gemini client) | ~150 MB | ~600 MB (during page rasterization) |
| **Total** | **~1.55 GB** | **~2.0 GB** |

Easily fits on the **$3/mo (4 GB)** Zeabur plan now that PaddleOCR/Surya are gone.

---

## Project Structure

```
ocr-pdf/
├── Dockerfile              # single-container build (API + Worker merged)
├── docker-compose.yml      # local dev — single service, no Redis
├── requirements.txt        # merged deps for API + Worker
├── config.yaml
├── store.py                # Redis / fakeredis provider (shared by API + Worker)
├── .env.example
├── .dockerignore
│
├── Api/
│   ├── main.py             # /api/upload, /api/start, …
│   └── static/
│       ├── index.html      # main UI
│       └── login.html
│
└── Worker/
    ├── worker.py           # job loop + per-page OCR caching
    ├── ocr_engine.py       # abstract OCREngine interface
    ├── engine_factory.py   # only "gemini" registered
    ├── gemini_engine.py    # ⭐ the Gemini API integration
    ├── pdf_ingestion.py    # PyMuPDF
    ├── structure_analysis.py # text → headings / paragraphs / footnotes / …
    └── pdf_assembly.py     # ReportLab / PyMuPDF: clean PDF output
```

---

## Troubleshooting

**Worker says `GEMINI_API_KEY environment variable is not set`:**
You forgot to add `GEMINI_API_KEY` to Zeabur env variables, or the value is empty. Check Zeabur UI → Environment Variables.

**Job fails with `Daily Gemini quota reached`:**
You've used all your free calls today. Wait until midnight Pacific Time (~UTC-7), or pause the job and resume tomorrow — cached pages will not be re-spent.

**429 errors in worker logs:**
The rate limiter should normally prevent this. If you see persistent 429s, your account might be on a more restrictive tier than the docs suggest — lower `rpm_limit` to 5 or 8 in `config.yaml`.

**Google OAuth callback error:**
Verify `BASE_URL` matches your Zeabur domain exactly (no trailing slash) and the redirect URI in Google Cloud Console is `BASE_URL + /auth/callback`.

---

## Cost Estimate

`gemini-2.5-flash-lite` on the **free tier** is 0¢ as long as you stay under 1,000 requests/day.

If you exceed the free tier and enable billing, paid pricing for `gemini-2.5-flash` is approximately **$0.30 per 1M input tokens** and **$2.50 per 1M output tokens**. A 100-page PDF at 400 DPI (downscaled to 2048px max side) uses roughly 200k input + 50k output tokens → about **$0.18 per 100 pages**.

---

## License

MIT
