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
pytest -q                                    # 87 tests, offline, ~3 s
```

## How it works

```
cook input ──► Pipeline 1 (tag request) ──► Pipeline 2 (rank library) ──► rules ──► 5 pages
             transformers.pipeline          transformers.pipeline        plain Python
```

| Layer | File | Job |
|---|---|---|
| Pipeline 1 | `same_table/tagging.py` | Sentence + chips → tags (steam / blanch / soup / stir, more-veg / more-meat, less-sweet, within-20-min). Uses zero-shot (base model) or text-classification (our fine-tune). A keyword lexicon always runs alongside it. |
| Pipeline 2 | `same_table/retrieval.py` | `feature-extraction` encoder. Embeds the ~100 library dishes once and ranks them by cosine similarity to the request. |
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

## Plug in your fine-tuned models (course rule: app model = notebook model)

After pushing the notebook fine-tunes to the Hub, set these as environment
variables or in `.streamlit/secrets.toml` (Streamlit Cloud: *Settings → Secrets*):

```toml
P1_MODEL_ID = "your-hf-user/same-table-tagger"     # multi-label text classifier
P1_TASK     = "text-classification"
P2_MODEL_ID = "your-hf-user/same-table-retriever"  # sentence encoder
P2_POOLING  = "mean"                               # "cls" for bge-style models
```

The P1 fine-tune's `id2label` must use these tag names: `steam, blanch, soup,
stir, more_veg, more_meat, less_sweet, quick`.

Until you set them, the defaults are `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
(zero-shot) and `BAAI/bge-small-zh-v1.5`.

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
