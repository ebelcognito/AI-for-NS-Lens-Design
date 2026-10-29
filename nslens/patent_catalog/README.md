# Automated folded-optics patent catalog

This pipeline searches for relevant publications, collects source documents,
expands a bounded citation graph, and builds an offline searchable gallery. It
runs without API credentials using the currently accessible Google Patents web
search endpoint. That endpoint is **not an official or stable API**; the collector
reports errors and retains progress instead of promising uninterrupted access.
For much larger production ingestion, use EPO OPS or a patent bulk dataset.

## Run

From the repository root, install the optional parser dependency if needed:

```bash
uv pip install --python .venv/bin/python 'beautifulsoup4>=4.12,<5'
# Alternatively: pip install -e '.[patents]'

.venv/bin/python -m nslens.patent_catalog run --max-new 60 --pdf-limit 3
```

Open `results/patent_catalog/index.html` directly in a browser. It requires no
server or internet access for filtering and locally downloaded previews. Patent
and source-PDF links naturally need the internet. No credentials are stored.

The defaults run 16 keyword/CPC searches, taking 10 results per query, seed five
known patents, and attempt at most 60 new document fetches. Citation expansion
is limited to one hop and two links each for cited, citing, and family-member
publications. These are request/queue budgets, not guaranteed useful-design counts.

```bash
# Continue the persisted queue without repeating searches:
.venv/bin/python -m nslens.patent_catalog collect --max-new 100

# Retry documents that failed earlier; use after the upstream service recovers:
.venv/bin/python -m nslens.patent_catalog collect --retry-errors --max-new 30 --delay 2

# Broaden discovery; cached search pages are reused unless refresh is requested:
.venv/bin/python -m nslens.patent_catalog discover --pages 3 --page-size 20

# Refresh live search results, then collect a bounded batch:
.venv/bin/python -m nslens.patent_catalog run --refresh-search --max-new 30

# Expand an additional citation hop; existing records are also reconsidered:
.venv/bin/python -m nslens.patent_catalog collect --max-depth 2 --max-new 100

# Import a Google Patents CSV export or one-ID-per-line file:
.venv/bin/python -m nslens.patent_catalog collect --import-csv patents.csv --max-new 30
.venv/bin/python -m nslens.patent_catalog collect --import-ids patents.txt --max-new 30

# No network requests:
.venv/bin/python -m nslens.patent_catalog render
.venv/bin/python -m nslens.patent_catalog status
```

Edit `searches.json` or pass `--config path/to/searches.json` to change categories,
queries, CPC filters, and seeds. Search discovery is independent of reconstruction:
the collector does not run optical simulations or alter existing experiments.

## Outputs

- `index.html`: text/category/score filters and optional provider-family grouping.
- `catalog.sqlite`: persistent queue, publication records, discovery provenance,
  citation/family edges, and per-invocation run manifests.
- `catalog.csv`, `catalog.jsonl`: portable publication-level exports.
- `discovered_candidates.csv`: every discovered publication, including pending
  and failed fetches, with title, source link, and collection status.
- `families.json`: publications grouped by the source's Google Patents family ID.
- `embodiment_references.json`: explicitly detected figure/embodiment references.
- `summary.json`, `last_run.json`: counts, status, errors, request/cache counts.
- `documents/<publication>/`: source HTML, extracted description, numerical table
  text, metadata/evidence record, preview when available, and optional source PDF.
- `cache/`: URL-keyed original response bytes and URL/time/content-hash metadata.

Source HTML is retained as archival evidence; use the generated escaped gallery
for browsing. A preview is the first available patent drawing, **not a verified
mapping to a particular optical embodiment**. The current version extracts HTML
text/tables; it does not OCR image-only tables or transcribe drawings. Optional
PDF downloads are limited to the top `--pdf-limit` ranked collected publications
and 30 MB per PDF. This budget applies to that run's ranked selection, not a
permanent collection-wide maximum as rankings change.

## What the ranking means

Every collected design is marked **unreviewed**. The 0–100 score adds evidence
for numerical tables (25), coefficients (20), positions/tilts (20), refractive
indices (10), curvature radii (10), aperture/pupil information (10), and reported
performance (5). Links and snippets let a reader inspect those signals.

`prescription_candidate` means numerical tables and coefficient-related text
were found. It does **not** mean a prescription is complete, consistent, linked
to a particular figure, or reproducible. Tables can be unrelated or incomplete;
polynomial conventions, clear apertures, and dispersion data may be missing.
`numerical_layout_candidate` and `layout_candidate` are weaker evidence tiers.

Category and physics tags are **document text mentions**, including possible
alternatives and background, not verified properties of every embodiment. The
pipeline does not infer exact fold counts or required polarization/diffraction
physics from a keyword. It retains explicit figure/embodiment associations as
unreviewed references; it does not equate their count with distinct designs.

Family grouping uses the provider's ID, without claiming it is a DOCDB simple
family or that every member has identical geometry. All collected publications
are retained. Different numerical embodiments still need inspection and later
separate model records. Our Figure 3A experiment is a separate spherical
approximation and does not automatically validate the collected patent.

## Network behavior and reproducibility

Requests are sequential with a one-second minimum start interval, a 25-second
timeout, bounded response sizes, and two retries for transient request failures.
HTTP 401/403 on document/search access or HTTP 429 stop collection immediately.
An unavailable individual preview/PDF is recorded and skipped without retrying
that denied URL or seeking an alternate route. Three consecutive document-service
failures pause the batch. Exit code 2 indicates a stopped/failed/interrupted run;
inspect `last_run.json`. Previously completed records survive and the gallery is
regenerated. Individual document errors remain visible in the queue; retry them
explicitly with `--retry-errors` after the service is available.

`--offline` permits cached requests only. `render` and `status` always operate
without network access. Normal repeat runs reuse caches and deduplicate publication
IDs and discovery edges. This initial tool is intended for **one writer process
per output directory**; simultaneous collectors in the same directory are not
supported. It does not install a scheduled job or start a persistent background
service. Each invocation is bounded and can be run by an external scheduler later.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p test_patent_catalog.py
```

Tests cover extraction/provenance, malformed responses, canonical IDs, family
and publication deduplication, resumability, HTML escaping, CSV imports, offline
rendering/cache reads, unavailable previews, and rate-limit/service-failure stops.

## Sources

- [Google Patents search syntax](https://support.google.com/faqs/answer/7049475)
- [Result grouping and CSV exports](https://support.google.com/faqs/answer/7049588)
- [EPO Open Patent Services](https://www.epo.org/en/searching-for-patents/data/web-services/ops)
- [Google public patent datasets](https://github.com/google/patents-public-data)
