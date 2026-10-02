# Unit specs: Waldstandsrevision Ilzertrift-Komplex 1878/90

One spec per TOC item of `[002] StALa, KdForsten A 383 I` that is linked to pages in the TOC UI
(`annotations_2026-09-28.json`, exported from the TOC UI on 2026-09-28). Format: [`../../SPEC.md`](../../SPEC.md).

The specs were written by Claude Code subagents from the transcripts and the 1200 px page images (I-16 and I-20 by
hand as worked examples) and checked with `spec-check`. Doubtful headers and readings are listed in each spec's `notes`
or form descriptions.

Restore them into freshly prepared units:

```bash
python -m ecological_ie prepare --dump … --toc-ui … --annotations ecological_ie/specs/ilzertrift_1878_90/annotations_2026-09-28.json \
    --specs ecological_ie/specs/ilzertrift_1878_90
```

After editing specs in `work/units`, copy them back with
`python -m ecological_ie export-specs --to ecological_ie/specs/ilzertrift_1878_90`.

Findings recorded in the specs: I-21 and I-00 read in volume order (the annotation order was a click order);
I-04 reads 1–9, 21–24, 10–20, 25–35; I-02 positions 33–48 are copies of positions 2–31 (`role: copy`);
I-12 holds only one filled page (the others are blank forms or an invalid draft); I-11 periods are 22-year periods
1883/1904 … 1993/2014; II-01 and I-02 areas are hectares with three decimals and yields Ster although the printed
headers say Tagwerk and Klafter; pages 197–199 of A 384 II (II-04, II-05) and the pages of I-13, I-14 and I-17
have no image in the dump.
