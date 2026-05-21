"""
worker.py — Required Changes
=============================
Two changes are needed in run_pipeline():

1. Update the import line to also import detect_dominant_language
2. After building structured_pages and toc, call detect_dominant_language()
   and pass it into DocumentStructure

Below are the exact before/after code blocks.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1: Update the import (inside run_pipeline)
# ─────────────────────────────────────────────────────────────────────────────

# BEFORE:
#     from structure_analysis import analyse_page, build_toc, DocumentStructure

# AFTER:
#     from structure_analysis import analyse_page, build_toc, DocumentStructure, detect_dominant_language


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 2: Detect language and pass it to DocumentStructure
# ─────────────────────────────────────────────────────────────────────────────

# BEFORE:
#         toc = build_toc(structured_pages)
#         structure = DocumentStructure(
#             title=ingested.meta.title, author=ingested.meta.author,
#             pages=structured_pages, toc=toc)

# AFTER:
#         toc = build_toc(structured_pages)
#         dominant_lang = detect_dominant_language(structured_pages)
#         logger.info(f"Detected dominant language: {dominant_lang}")
#         structure = DocumentStructure(
#             title=ingested.meta.title, author=ingested.meta.author,
#             pages=structured_pages, toc=toc,
#             dominant_language=dominant_lang)
