# Same Table / 糖友同枱

A Streamlit app that helps a family cook plan **one shared home dinner** for a
diabetes patient and the household. The dishes stay the same for everyone. Only
the patient's plate changes, using three simple rules. This is a classroom
prototype for the Hospital Authority. It is not a prescription and is not
connected to HA Go.

## Run locally

```bash
pip install -r requirements-dev.txt
streamlit run app.py                         # downloads the two HF models on first run
SAME_TABLE_OFFLINE=1 streamlit run app.py    # no downloads: keyword + char-gram fallback
pytest -q                                    # 101 tests, offline, ~4 s
```

## How it works

```
cook input ──► Pipeline 1 (tag request) ──► Pipeline 2 (rank library) ──► rules ──► 5 pages
             transformers.pipeline          transformers.pipeline        plain Python
```

| Layer | File | Job |
|---|---|---|
| Pipeline 1 | `same_table/tagging.py` | Sentence + chips → tags (steam / blanch / soup / stir, more-veg / more-meat, less-sweet, within-20-min). Uses zero-shot (base model) or text-classification (our fine-tune). A keyword lexicon always runs alongside it. |
| Pipeline 2 | `same_table/retrieval.py` | `feature-extraction` encoder. Embeds the ~100 library dishes once and ranks them by cosine similarity to the request plus a weighted exact-match score (hybrid). |
| Rules | `same_table/rules.py` | Dish-count table (1→2 … 6→8, 7+ → 8); category split; 14-day log with non-repeat; 20-min filter; three fixed patient actions; shopping-list merge that never invents ingredients. |
| Planner | `same_table/planner.py` | Glue code, and the single function the app calls. |
| Library | `same_table/library.py`, `data/` | Validated dish rows. `seed_library.json` holds 30 hand-written placeholder rows. |
| App | `app.py` | Tonight → Menu → How to cook → How to plate → List, with a zh/EN toggle and a disclaimer on every page. |

**Fallback plan (error handling):** if the library is too small to fill the menu,
the rules relax one at a time, in this order: allow repeats from the last 14
days, drop the 20-min filter, then borrow from another category. Each
relaxation shows a note on the Menu page. If a model fails to load, the app
still works in basic mode (keywords + char TF-IDF), and the footer says so
without naming the model.

## Train and evaluate the models

```bash
python scripts/make_training_data.py              # data/train/*.jsonl
# then run, in this order (Colab: Runtime -> Restart & Run All):
#   notebooks/Model1_finetune_tagger.ipynb        fine-tunes 3 candidates for Pipeline 1, picks one, uploads it
#   notebooks/Model2_experiments.ipynb            tests 3 encoders for Pipeline 2, tests the whole app, writes the Excel file
python scripts/make_submission.py --group 01 --leader 07123456    # Canvas folder layout + zip
```

Pipeline 1 is fine-tuned (it is the pipeline closest to the objective). Pipeline 2 is not fine-tuned:
three pre-trained encoders are tested on our own queries. Results of the committed run (CPU):

| Pipeline 1: 8 tags from a sentence | Accuracy, `trainer.evaluate()` (600 held-out-phrase sentences) | Accuracy, 108 hand-written test | CPU sec / 100 requests |
|---|---|---|---|
| Keyword lexicon, no model | 0.40 | 0.65 | <0.01 |
| Zero-shot mDeBERTa, no training | n/a | 0.29 | 93 |
| `hfl/rbt3`, fine-tuned | 0.58 | 0.85 | 0.9 |
| **`hfl/rbt6`, fine-tuned (picked)** | **0.64** | **0.89** | 1.8 |
| `hfl/chinese-roberta-wwm-ext`, fine-tuned | 0.65 | 0.81 | 4.3 |

Accuracy is strict: all 8 tags of a sentence must be right. Tag-level accuracy of the picked model is 0.93
(evaluation) and 0.98 (test).

| Pipeline 2: dish retrieval, hybrid score | Top-5 accuracy (200 literal test queries) | Precision@5 (24 paraphrases) | CPU sec / 100 queries |
|---|---|---|---|
| Character TF-IDF, no model | 0.890 | 0.48 | <0.1 |
| **`BAAI/bge-small-zh-v1.5` (picked, w = 3)** | 0.895 | 0.54 | 2.3 |
| `shibing624/text2vec-base-chinese` (w = 2) | 0.895 | 0.58 | 5.9 |
| `paraphrase-multilingual-MiniLM-L12-v2` (w = 5) | 0.895 | 0.60 | 2.7 |

Whole app (fine-tuned tagger + keyword lexicon → retriever → rules): 18 of 20 dinner requests pass every
objective check; about 0.03 s per request on CPU.

Model choice rule for both pipelines: highest accuracy; models within one standard error of the best are
tied and the shortest run time wins.

Known limits: training sentences are template-generated (the evaluation split uses held-out phrases and
the test set is hand-written); without fine-tuning, the encoder only matches plain character matching on
literal ingredient queries and its gain is on paraphrases; the paraphrase set is small (24).

## Settings for the deployed app (course rule: app model = notebook model)

After `Model1_finetune_tagger.ipynb` uploads the tagger, set these in Streamlit Cloud → *Settings → Secrets*
(or `.streamlit/secrets.toml` locally):

```toml
P1_MODEL_ID = "your-hf-user/same-table-tagger"     # the fine-tuned multi-label classifier
P1_TASK     = "text-classification"
```

Pipeline 2 needs no secret: `BAAI/bge-small-zh-v1.5`, CLS pooling and hybrid weight 3 are the defaults in
`same_table/config.py` (override with `P2_MODEL_ID`, `P2_POOLING`, `P2_HYBRID_WEIGHT`). Until `P1_MODEL_ID`
is set, Pipeline 1 falls back to the zero-shot model, which is far less accurate.

## English mode

The interface text is in `same_table/i18n.py`. Recipe text (dish names, ingredients, amounts, steps) comes
from `data/library_en.json` (and `data/seed_library_en.json`), keyed by dish id. These translations were
prepared offline with AI assistance; small Hugging Face translation models (`Helsinki-NLP/opus-mt-zh-en`,
`facebook/nllb-200-distilled-600M`) were tried first and were not usable for cooking terms. No translation
model runs inside the app. A translation is used only when its ingredient and step counts match the Chinese
row; otherwise that dish shows in Chinese with a note. English dish names are also added to each dish's
search text, so English requests can find dishes. Spot-check the translations before submission, and
add rows to `library_en.json` for any dish you add to the library.

## Build the real library (~100 dishes)

```bash
python scripts/build_library.py export   # HF dataset -> data/library_review.csv (auto-suggested labels)
# review category / method in the CSV, set keep=0 to drop rows
python scripts/build_library.py import   # -> data/library.json + data/covers/*.jpg (the app uses them automatically)
python scripts/build_library.py covers   # re-try photo downloads only
```

**Photos:** the dataset's `image` links use iCook's retired resize proxy. That proxy
only redirects now, and the original storage address returns 403. The script pulls
the original address out of each link, rebuilds it on iCook's current image
server, and saves a 600 px copy to `data/covers/<id>.jpg`. The app shows the
local copy first, then the URL, then a category icon. Every photo is captioned
"image from the library, not generated". Copyright stays with the iCook authors.

**Review checklist for `library_review.csv`:** the auto-labels come from keyword
rules, so check every row. Fix `category` where a few dried shrimp or a meat garnish
tipped a vegetable dish into seafood or meat. Set `keep=0` for snacks, noodle and
rice dishes, and photos that don't show the dish. The dataset has few steamed
dishes, so hand-pick extra 蒸 rows if you want the Steam chip to have choices.

## Deploy (Streamlit Community Cloud)

1. Push to a **public** GitHub repo.
2. On share.streamlit.io, create a new app with main file `app.py`.
3. Add the secrets above.

The 14-day log is seeded demo data kept in the session only, because Streamlit
Cloud has no database.

## Out of scope

Checkout, prices, HA Go, grams, glucose, kcal, camera, OpenAI or other non-HF
APIs, "suitable for diabetes" badges.
