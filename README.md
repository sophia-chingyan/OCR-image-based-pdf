# PDF/JPG → Clean PDF Converter

Self-hosted, single-user web app that converts **image-based PDF files and JPG images** to clean, re-typeset PDF files using **Google Gemini** for OCR.

- ✅ Upload **PDF or JPG/JPEG** files — a JPG is auto-wrapped into a one-page PDF and OCR'd the same way
- ✅ OCR via Google Gemini (`gemini-3.5-flash-lite` by default)
- ✅ Languages: Traditional Chinese, Simplified Chinese, Japanese, Korean, English (and 100+ others)
- ✅ Auto-detects horizontal / vertical text layout per page
- ✅ Clean PDF with correct CJK font/CMap per detected language
- ✅ Re-embeds images, preserves hyperlinks, headings, TOC, footnotes, page numbers
- ✅ Async job queue with Start / Pause / Stop / Delete / Retry controls
- ✅ Google OAuth2 authentication (single-user, allowlist by email)
- ✅ One Gemini API call per page (efficient, low cost / quota)
- ✅ Per-page OCR caching — pause/resume without spending extra quota

---

## Why Gemini?

The previous PaddleOCR / Surya implementations needed too much RAM for the Zeabur server. Gemini moves OCR off-server entirely — the worker just sends each page image to Google's API and receives structured JSON back. The Zeabur worker now uses **under 1 GB RAM** and needs no GPU or PyTorch.

Trade-off: each page = 1 Gemini API call, so **daily free-tier quota matters**. If you pause or a job fails partway through, the OCR results for completed pages are cached — resuming costs zero extra quota for those pages.

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

The `config.yaml` ships with **paid-plan** rate limits (`rpm_limit: 2000`, `rpd_limit: 10000`).
If you are on the **free tier**, lower these values to stay within quota. Free-tier RPM/RPD limits
change over time and by account — check your current limits at
[Google AI Studio → Rate limits](https://aistudio.google.com) before deploying, and set
`rpm_limit` / `rpd_limit` in `config.yaml` accordingly.

The quota resets at midnight Pacific Time. Each PDF page (or each uploaded JPG) = 1 request.

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
4. Upload a PDF or JPG, click **Start**, watch progress
5. When done, click **↓ Clean PDF** to download

---

## Configuration (`config.yaml`)

```yaml
ocr:
  engine: gemini
  model_name: "gemini-3.5-flash-lite"   # default; change to gemini-3.5-flash for higher accuracy
  rpm_limit: 2000                        # paid plan; lower to your free-tier RPM
  rpd_limit: 10000                       # paid plan; lower to your free-tier RPD
  max_retries: 5
  request_timeout_s: 180
  confidence_threshold: 0.7
  dpi: 300                               # rasterization DPI (300 balances quality vs memory)

pipeline:
  max_pdf_size_mb: 100
  page_batch_size: 10
  upload_retention_hours: 24
  output_retention_days: 7
  tmp_cleanup_on_complete: true

server:
  max_concurrent_jobs: 1
  port: 8080
  job_history_limit: 10
```

### Free tier overrides

If you are on the free Gemini tier, lower `rpm_limit` / `rpd_limit` in `config.yaml` to match the
current free-tier limits shown for your account at
[Google AI Studio → Rate limits](https://aistudio.google.com), e.g.:

```yaml
ocr:
  rpm_limit: 15
  rpd_limit: 1500
```

### Switching to `gemini-3.5-flash` (higher accuracy)

```yaml
ocr:
  model_name: gemini-3.5-flash
  rpm_limit: 2000    # paid plan; lower to your free-tier RPM
  rpd_limit: 10000   # paid plan; lower to your free-tier RPD
```

Then `docker compose restart app` to apply.

---

## Memory Budget

| Component | Idle | Peak |
|---|---|---|
| OS + existing services | ~1.2 GB | ~1.2 GB |
| FastAPI + in-process store | ~200 MB | ~200 MB |
| Worker thread (Gemini client) | ~150 MB | ~400 MB (during page rasterization at 300 DPI) |
| **Total** | **~1.55 GB** | **~1.8 GB** |

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
    ├── pdf_ingestion.py    # PyMuPDF + JPG→1-page-PDF conversion
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

`gemini-3.5-flash-lite` on the **free tier** is 0¢ as long as you stay under your account's daily request quota. A JPG upload costs exactly 1 request (it is OCR'd as a single-page PDF).

If you exceed the free tier and enable billing, check current per-token pricing for `gemini-3.5-flash-lite` / `gemini-3.5-flash` at [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing) — pricing and quotas are updated by Google independently of this project.

---

## License

MIT
