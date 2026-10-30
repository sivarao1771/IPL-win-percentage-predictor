# IPL Win Probability Predictor

Predicts the probability that the chasing team wins an IPL match, from the state of the second innings: target, score, legal balls bowled and wickets lost.

- **Model:** gradient-boosted trees (scikit-learn), trained on the 2008-2024 seasons and tested once on 2025-2026.
- **Ways to use it:** a Python function and command line ([`ml/predict.py`](ml/predict.py)), or a NestJS API ([`api/`](api/)). Both run the same exported ONNX model and give identical answers.
- **Honest evaluation:** split by season, so no match is in both training and test, compared against simple baselines, and every number reported exactly as measured.

An earlier version of this project reported **97.85% accuracy**. That number came from data leakage (explained [below](#why-the-old-9785-was-wrong)). The honest test accuracy is **77.46%**.

## Results (test seasons 2025-2026, 138 matches)

All models were trained on 2008-2024 and scored once on 2025-2026. Brackets show 95% confidence intervals from 1,000 resamples of whole test matches.

| Model | Accuracy | Log loss | Brier score |
|---|---|---|---|
| B0: always predict the training base rate | 55.94% (47.06-64.06%) | 0.6891 (0.6824-0.6964) | 0.2480 (0.2446-0.2516) |
| B1: logistic regression on runs left, balls left, wickets left | 78.35% (73.71-83.16%) | 0.4695 (0.3955-0.5536) | 0.1515 (0.1247-0.1789) |
| **Final model:** gradient-boosted trees | 77.46% (72.11-82.53%) | 0.4846 (0.3895-0.5934) | 0.1561 (0.1235-0.1922) |

**What these metrics mean.** Accuracy: how often the side of 50% matched the result. Log loss: penalty for the probability given to what actually happened (lower is better; a constant 50% guess scores 0.6931). Brier score: mean squared gap between probability and result (lower is better; a constant 50% guess scores 0.25). Log loss is the main metric because the model's job is to output probabilities, not just pick a side.

**Plain summary.** On the test seasons the final model **did not beat** the simple logistic baseline: log loss 0.4846 vs 0.4695. The difference is within the noise: the paired 95% interval for (final - B1) log loss, resampling the same test matches for both, is -0.0389 to +0.0743, which includes 0. On the 2024 validation season, used to choose the model, the trees were ahead (0.3829 vs 0.4171). The trees were kept because the selection rule was fixed before the test seasons were opened; switching to B1 after seeing the test results would be choosing on the test set.

**By phase of the chase (test, log loss, lower is better):**

| Phase | Rows | Final model | B1 logistic |
|---|---|---|---|
| Overs 1-6 | 5,341 | 0.6349 | 0.6478 |
| Overs 7-15 | 7,595 | 0.4714 | 0.3989 |
| Overs 16-20 | 2,978 | **0.2487** | 0.3296 |

The trees are clearly better at the end of the chase (overs 16-20: accuracy 87.31% vs 86.00%, Brier 0.0799 vs 0.1008) and worse in the middle overs.

The full report, including the model-selection grid, calibration tables and confidence intervals, is in [`ml/reports/results.md`](ml/reports/results.md).

## Why the old 97.85% was wrong

The old notebook split the data **randomly by row**. Each row is one ball, and a chase has about 120 of them, so every match ended up with roughly 96 balls in training and 24 in testing. Rows from the same match have nearly identical features and the same result, and team + city + target almost uniquely identifies a match (1,228 distinct combinations across 1,237 chases). A model with enough capacity can recognise the match instead of predicting it. The 97.85% was also the Random Forest's out-of-bag score, which has the same problem.

Re-running both evaluation styles on the current data shows the effect:

| Test rows picked by | Model | Accuracy | Log loss |
|---|---|---|---|
| Random rows (old method) | Old notebook Random Forest + team/city | 95.73% | 0.2166 |
| Random rows (old method) | Final model settings + team/city | 82.35% | 0.3753 |
| Random matches | Final model settings | 77.57% | 0.4422 |
| **Seasons (this project)** | **Final model** | **77.46%** | **0.4846** |

The deep forest (150 trees, depth 20) exploits the leak; the small final model mostly can't. Splitting by season also tests what matters in practice: predicting matches from seasons the model has never seen.

## Method

**Data.** `IPL.csv`, a ball-by-ball IPL dataset: 295,732 rows, 1,243 matches, 2008-04-18 to 2026-05-31. The original notebook loaded it from the Kaggle dataset `ipl-dataset2008-2025`; the copy used here also includes the 2026 season. The exact file is identified by its SHA-256 (`425c40c4eb60…`, full value in [`ml/reports/data_summary.json`](ml/reports/data_summary.json)). The dataset is not in this repository.

**Cleaning** ([`ml/prepare_data.py`](ml/prepare_data.py)). Renamed franchises are merged (e.g. Delhi Daredevils → Delhi Capitals). 58 matches are excluded, each with a logged reason: D/L (23), tie (16), no result (9), first innings shortened by rain (8, detected from how the innings ended) and 2 shortened matches the rules can't detect, excluded by hand with a documented reason. 1,185 matches remain.

**Rows.** One row per chase state: the state after every delivery, plus the state before the first ball. States where the result is already known (target reached, no balls or wickets left) are removed. Label: 1 if the chasing team won.

**Features** ([`ml/features.py`](ml/features.py)), all known at the moment of prediction: runs left, balls left, wickets left, target, current run rate, required run rate. Team and city were tested and made validation results worse (log loss 0.4198 vs 0.3829), so they are not used.

**Split by season.** No match appears in two splits.

| Split | Seasons | Matches | Rows | Chases won |
|---|---|---|---|---|
| Train | 2008-2023 | 977 | 113,871 | 54.25% |
| Validation | 2024 | 70 | 8,108 | 51.43% |
| Test | 2025-2026 | 138 | 15,914 | 57.97% |

**Model selection** ([`ml/train.py`](ml/train.py), 2024 only). A grid of 72 gradient-boosted tree settings: learning rate, number of trees, leaves per tree, and with or without monotonic constraints. The constraints force cricket logic: more runs needed or a higher required rate can only lower the win probability; more balls or wickets left can only raise it. The selection rule was fixed before testing: keep settings within 1% of the lowest validation log loss, then prefer monotonic, then the smallest (trees × leaves), then the higher learning rate. Chosen: 50 trees, 15 leaves, learning rate 0.1, monotonic. One season of 70 matches can't separate most settings, which is why the rule picks among near-ties instead of trusting the single lowest score.

**Final evaluation** ([`ml/evaluate.py`](ml/evaluate.py)). The chosen settings and both baselines are retrained on 2008-2024 and scored once on 2025-2026.

**Export** ([`ml/export_onnx.py`](ml/export_onnx.py)). The final model is exported to ONNX ([`ml/artifacts/model.onnx`](ml/artifacts/model.onnx), version `ipl-chase-hgb-d81f5e586f86`). The export fails if any probability differs from scikit-learn by more than 1e-5; the largest observed gap is 2.055e-7 across test, validation and about 1.75 million edge states. [`model_meta.json`](ml/artifacts/model_meta.json) records the feature order, the input rules with their error messages, and the model's SHA-256, which both the Python function and the API check before using the model.

## Limitations

- **The final model is not better than the logistic baseline on the test seasons.** The difference is within the noise; 138 test matches is not enough to separate them.
- **Recent seasons are under-predicted.** On both 2024 and 2025-2026 the chasing team won more often than the model predicted (in 9 of 10 probability buckets on test; for example predictions of 40-50% won 74.96% of the time). A possible cause is the Impact Player rule introduced in 2023, which effectively lengthens the batting order. This is an untested hypothesis.
- **Hopeless chases don't go near 0%.** 100 needed off 6 balls gets **4.828%**. Each tree leaf averages at least 100 training rows and such states are rare, so the model can't output values close to 0 or 1 at the extremes.
- **Targets outside the training range (68-288)** get the model's nearest learned values; the trees don't extrapolate. Inputs are limited to targets of 1-300.
- **Not covered:** rain-shortened (D/L) matches, ties and super overs, and match context such as pitch, toss or players.
- **The test seasons are used up.** Any change to the model must be judged on a new season (2027 or later), not on 2025-2026 again.

## How to run

### Python: function and command line

Requires Python 3.11. From the repository root:

```bash
python -m venv ml/.venv
ml/.venv/Scripts/python.exe -m pip install -r ml/requirements.txt    # Windows
# ml/.venv/bin/python -m pip install -r ml/requirements.txt          # macOS / Linux
```

Command line:

```bash
ml/.venv/Scripts/python.exe ml/predict.py --target 180 --score 95 --balls 66 --wickets 3
```
```
Chasing 180: 95/3 after 66 balls -> need 85 off 54, 7 wickets left
Win probability: 63.56%  (model ipl-chase-hgb-d81f5e586f86)
```

Add `--json` for machine-readable output. Invalid input prints every problem and exits with code 2:

```bash
ml/.venv/Scripts/python.exe ml/predict.py --target 301 --score 95.5 --balls 66 --wickets 10
```
```
error: target must be between 1 and 300
error: score must be an integer
error: wicketsLost must be between 0 and 9
```

As a function (run from the `ml/` folder, or add it to `sys.path`):

```python
from predict import predict_win_probability, InvalidInputError

predict_win_probability(target=180, score=95, balls_bowled=66, wickets_lost=3)
# 0.6356436014175415
```

The model is loaded once, on the first call. Invalid input raises `InvalidInputError` (a `ValueError`) whose `.errors` lists every problem.

### API (NestJS)

Requires Node.js 22. From `api/`:

```bash
npm ci
npm run build
npm start            # http://localhost:3000
```

| Endpoint | Description |
|---|---|
| `POST /api/predict` | Win probability for a match state |
| `GET /api/health` | `{ "status": "ok", "modelVersion": "..." }` |
| `GET /docs` | Swagger UI (OpenAPI JSON at `/docs-json`) |

```http
POST /api/predict
{"target": 180, "score": 95, "ballsBowled": 66, "wicketsLost": 3}

200 OK
{"winProbability": 0.6356436014175415,
 "modelVersion": "ipl-chase-hgb-d81f5e586f86",
 "features": {"runsLeft": 85, "ballsLeft": 54, "wicketsLeft": 7,
              "crr": 8.636363636363637, "rrr": 9.444444444444445}}
```

Invalid input returns `400` with one message per invalid field (type errors before range errors; unknown fields are rejected):

```json
{"statusCode": 400, "error": "Bad Request",
 "message": ["target must be between 1 and 300", "score must be an integer"]}
```

Environment variables: `PORT` (default 3000), `MODEL_DIR` (default `../ml/artifacts`), `CORS_ORIGIN` (comma-separated origins; CORS is off when unset). The API refuses to start if the model files are missing or don't match their SHA-256.

### Input rules (both interfaces)

| Field | Allowed values |
|---|---|
| `target` | Integer, 1-300 (first-innings score + 1) |
| `score` | Integer, 0 to target - 1 |
| `ballsBowled` / `--balls` | Integer, 0-119 legal balls (wides and no-balls don't count) |
| `wicketsLost` / `--wickets` | Integer, 0-9 |

### Tests

```bash
ml/.venv/Scripts/python.exe -m pytest ml/tests    # Python: 85 tests
cd api && npm test                                 # API: 256 tests
```

The tests need no dataset: they use small synthetic matches and the committed model files. Both test suites check the same 200 parity cases exported from Python, so the Python function and the API are proven to compute identical features and probabilities. GitHub Actions runs both on every push ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

### Reproducing the model and results

Requires `IPL.csv` at `data/IPL.csv` (check its SHA-256 against `data_summary.json`). From the repository root:

```bash
ml/.venv/Scripts/python.exe ml/prepare_data.py   # clean, build chase states, split by season
ml/.venv/Scripts/python.exe ml/train.py          # choose settings on 2024
ml/.venv/Scripts/python.exe ml/evaluate.py       # final test, writes ml/reports/results.md
ml/.venv/Scripts/python.exe ml/export_onnx.py    # export and parity check
```

All steps are deterministic (fixed seed 42, pinned package versions in [`ml/requirements.txt`](ml/requirements.txt)).

## Project structure

```
ml/
  features.py          feature definitions shared by every step
  prepare_data.py      cleaning, exclusions, season split
  models.py            baselines and gradient-boosted trees
  metrics.py           accuracy, log loss, Brier, calibration, bootstrap
  train.py             model selection on 2024
  evaluate.py          one-time test on 2025-2026, report
  export_onnx.py       ONNX export and parity check
  predict.py           prediction function and command line
  artifacts/           model.onnx, model_meta.json, parity_cases.json
  reports/             results.md, metrics.json, data_summary.json, excluded_matches.csv
  tests/               pytest
api/
  src/                 NestJS app: model loading, validation, /api/predict, /api/health, Swagger
  test/                end-to-end tests
ipl-win-predic.ipynb   the original notebook, kept as history (its results are not valid; see above)
```

## Future work

- **Recency weighting:** give recent seasons more weight in training, to test whether the under-prediction of recent chases comes from the game changing (for example the Impact Player rule).
- **Multi-season (rolling) validation:** choose settings on several recent seasons (about 400 matches) instead of 2024 alone (70), so differences between settings become measurable.
- Both must be judged on a new, untouched test season.
