# KANSER OOF Discovery Report

Status: **VERIFIED_ROW_LEVEL_OOF_FOUND**

The owner final ZIP contains a 388-row repeated, group-safe OOF mean-score file.
Its Variant_ID order and development labels match the supplied raw training CSV exactly.
All scores are finite, and every locked prediction matches the frozen model threshold.
The confusion counts (TN=114, FP=6, FN=96, TP=172) match the owner RESULTS.json record.

`KANSER_ROW_LEVEL_OOF.csv` is byte-identical to the owner's
`08_selected_oof_predictions.csv` and was copied without score rounding or aggregation.

The file uses development labels only. It contains no final/test label and does not
establish clinical utility. No model, preprocessing, threshold, or review policy was changed.

Per-repeat row-level scores are not present in the ZIP; the verified file is the owner-frozen
mean repeated OOF score per row. See `KANSER_OOF_PROVENANCE.json` for hashes and checks.
