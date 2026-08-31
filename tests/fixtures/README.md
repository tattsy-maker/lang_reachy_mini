# Test fixtures

## Face photos (`faces/`)

Six public-domain NASA portraits (US government works), downloaded from
Wikimedia Commons at 1280px width on 2026-08-31. Three distinct people, two
photos each — which gives T2 same-person pairs *and* three different-person
pairs to measure thresholds against.

| File | Person | Source (Wikimedia Commons) | Notes |
|---|---|---|---|
| `sunita_a.jpg` | Sunita Williams | `File:Sunita Williams.jpg` | early-career official portrait |
| `sunita_b.jpg` | Sunita Williams | `File:Sunita Williams in 2018 (cropped).jpg` | 2018 portrait — the cross-year same-person pair |
| `tracy_a.jpg` | Tracy Caldwell Dyson | `File:Tracy Dyson portrait 2023.jpg` | Jan 2023 |
| `tracy_b.jpg` | Tracy Caldwell Dyson | `File:NASA astronaut Tracy Dyson poses for a portrait.jpg` | same Jan 2023 session, different shot |
| `scott_a.jpg` | Scott Kelly | `File:Scott J. Kelly.jpg` | 2014 EMU-suit portrait |
| `scott_b.jpg` | Scott Kelly | `File:Scott Kelly, Johnson Space Center portrait, 2019 (cropped).tif` (JPEG render) | 2019 portrait — a second cross-year pair |

All six are tagged **Public domain** on Commons (NASA photographs). The
family's real enrollment photos never live in this repo; these stand in for
"person A/B/C" wherever tests need faces.

## Video (`video/sunita_clip.avi`)

A 6-second, 640x480, 15 fps webcam-style clip of `sunita_b.jpg` —
synthesized (slow pan/zoom, exposure flicker, sensor noise) by
[make_video.py](make_video.py), because we have no licensable real webcam
footage. It is committed; rerun the script only to change it. If a task
later needs footage with real head motion, record it then and note it here.

## Learner tree (`learners/`)

A sample of the T3 store layout from the spec (section 4B): `maria/` is a
`family`-tier intermediate-Spanish learner with two sessions of notes
(newest first), `sam/` is a `guest`-tier beginner-French learner with one.
The `embedding` values are placeholder numbers, not real face embeddings —
T2/T3 tests that care about real embeddings compute their own from
`faces/`.
