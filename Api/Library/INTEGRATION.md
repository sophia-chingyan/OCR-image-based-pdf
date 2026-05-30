# Integration notes

Drop the four HTML files into `Api/static/` (replacing the existing
`index.html` and `login.html`, adding `library.html` and `login_error.html`),
then make the small `Api/main.py` change below.

```
Api/static/
├── index.html         # replaced — redesigned, job pipeline unchanged
├── library.html       # NEW — download / delete completed outputs
├── login.html         # replaced — restyled
└── login_error.html   # NEW (optional) — restyled access-denied page
```

No new API endpoints are required. The Library page is rendered entirely on
the client from the existing endpoints:

* `GET    /api/history`              → lists jobs (Library shows `status === "done"`)
* `GET    /api/download/{id}/clean`  → per-row download
* `DELETE /api/delete/{id}`          → per-row / bulk / delete-all

---

## 1. Required — add the `/library` route

`library.html` is the authenticated UI, so it must be served the same way as
`index.html` (never via a public StaticFiles mount — that was deliberately
disabled in the original code). Add this route to `Api/main.py`, right after
the existing `@app.get("/")` handler:

```python
@app.get("/library", response_class=HTMLResponse)
async def library(request: Request):
    user = await get_current_user(request)
    if not user:
        # Not signed in → bounce to the login page rendered at "/"
        return RedirectResponse(url="/", status_code=302)
    library_path = _static_dir / "library.html"
    async with aiofiles.open(library_path) as f:
        return HTMLResponse(await f.read())
```

That is the only change needed for full functionality.

---

## 2. Optional — use the styled access-denied page

The current callback returns a bare `<h1>403 Access Denied</h1>`. To render
the new `login_error.html` instead, replace the 403 branch inside
`auth_callback` in `Api/main.py`:

```python
    if email != ALLOWED_EMAIL:
        error_path = _static_dir / "login_error.html"
        try:
            async with aiofiles.open(error_path) as f:
                html = (await f.read()).replace(
                    "{{ error }}",
                    "This Google account is not authorized to use this app.",
                )
            return HTMLResponse(html, status_code=403)
        except Exception:
            return HTMLResponse("<h1>403 Access Denied</h1>", status_code=403)
```

(`{{ error }}` is just a placeholder string in the template — swapped in with a
plain `str.replace`, so no template engine is added.)

---

## What changed in the UI

* **Design system** — warm paper background, Lora (serif headings) + DM Sans
  (body), terracotta accent, dark sticky header with Convert / Library / Logout
  navigation. Applied consistently across all pages.
* **`index.html` (Convert)** — same look, **identical job logic**: drag-drop
  upload, language-priority picker, Start / Pause / Stop / Delete, 5-second
  status polling, the live "OCR Processing" working indicator, progress bar,
  and job-history cards (now showing a status badge) are all preserved.
* **`library.html` (Library)** — table of completed conversions with select-all,
  bulk "Delete Selected", "Delete All", per-row Download / Delete, a confirm
  modal, and toast notifications. Pruned outputs show an "Expired / Unavailable"
  state but can still be deleted.
