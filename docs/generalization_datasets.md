# Candidate datasets for the generalizability test

Per the priority realignment (`docs/DECISIONS.md`, 2026-09-01): once accuracy on
GraSP is strong, validate on a second, independent surgical instrument dataset.
Researched via web search 2026-09-01, not yet acted on.

## Primary recommendation: EndoVis 2017 / 2018 (MICCAI Endoscopic Vision Challenge)

- Same robotic platform family as GraSP (da Vinci), overlapping instrument
  taxonomy (Bipolar Forceps, Prograsp Forceps, Large Needle Driver, etc. --
  close to GraSP's own 7 classes).
- The standard, most-cited benchmark in exactly this subfield -- a result here
  is instantly comparable against a large existing body of published work.
- EndoVis 2017: 10 videos, da Vinci. EndoVis 2018: 15 sequences, 7 predefined
  instrument categories.
- Access: requires registering via the MICCAI challenge / synapse.org, not
  an open direct download.

## Secondary option: CholecInstanceSeg (Scientific Data, 2025)

- Laparoscopic (not robotic) -- a genuinely more different domain than
  EndoVis, a better test of broad generalization rather than cross-hospital/
  cross-patient variation within the same robotic platform.
- Largest open-access tool instance-segmentation dataset to date: 41.9k
  annotated frames, 64.4k tool instances, 85 clinical procedures.
- Different instrument taxonomy (laparoscopic tools, not GraSP's 7 classes)
  -- a harder, more genuine generalization stress test, at the cost of less
  direct comparability.

## Considered, not recommended

- **Spine Endoscopic Atlas (SEA)** -- spinal endoscopic surgery, 4,851 images,
  publicly available (figshare). Instrument set too different from GraSP's
  (specialized spine tools) for a meaningful class-transfer test.
- **ROBUST-MIPS / ROBUST-MIS** -- laparoscopic instrument segmentation + pose,
  similar niche to CholecInstanceSeg but smaller and older; CholecInstanceSeg
  supersedes it as the more current choice in that same category.

## Open question, not yet decided

Whether to test zero-shot transfer (train on GraSP only, evaluate directly on
the new dataset) or fine-tune on the new dataset's own training split -- these
answer different questions ("does GraSP-only training generalize at all" vs.
"does the architecture/approach generalize given some new-domain data") and
should be decided deliberately, not defaulted into.
