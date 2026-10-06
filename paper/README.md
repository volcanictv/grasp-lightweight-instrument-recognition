# MIDL 2027 paper (main track, full paper)

Deadlines (23:59 AoE): abstract registration 2026-11-30, full paper 2026-12-04. Limit: 10 pages excluding references, acknowledgements and appendices. Single-blind.

Build: upload this folder to Overleaf (MIDL template), or locally with tectonic:

    tectonic -X compile main.tex --outdir build

Layout: `main.tex` includes `sections/*.tex`; figures are in `figures/` (PNG made by `scripts/build_pi_report_docx.py`; regenerate as vector PDF before submission); `references.bib` entries marked "verify" were written from memory.

Every number comes from `docs/reports/gtbox_sam/*.json`, `docs/reports/latency/*.json` and `docs/reports/realtime/*.json`. Every open item is a red `\todo{...}`: search for `todo{` and leave none before submitting.
