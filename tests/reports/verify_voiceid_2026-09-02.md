# Speaker verification gate — 2026-09-02

Model `speechbrain/spkrec-ecapa-voxceleb` on `cuda:0`; 20 clips, speakers: af_heart, am_adam, bf_emma, bm_george, ef_dora.

| Pairs | n | min | mean | max |
|---|---|---|---|---|
| same_speaker | 30 | 0.717 | 0.8204 | 0.8975 |
| different_speaker | 160 | -0.0278 | 0.1331 | 0.3909 |
| holdout_same | 20 | 0.7859 | 0.8742 | 0.9288 |
| holdout_different | 80 | -0.0121 | 0.1449 | 0.3299 |

Thresholds: accept ≥ 0.6, reject < 0.45. Band is CLEAN.
