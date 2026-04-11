# IS450 Text Mining — Bangkok Airbnb Review Analysis

Analysis of **380,000+ Bangkok Airbnb guest reviews** to extract sentiment, topics and aspect-level insights for hosts.

---

## Code & Notebooks

All analysis notebooks are in this repository under `Actual Work/`. They are organised by task:

| Folder | Contents |
|---|---|
| `pre-processing/` | Cleaning, language detection, NLP pipeline |
| `features/` | TF-IDF, Word2Vec, TextCNN feature engineering |
| `task1-sentiments/` | VADER labelling, Logistic Regression vs RoBERTa |
| `task2-topics/` | LDA topic modelling, BERTopic with Gemini labels |
| `task3-combination/` | Rule-based ABSA (NSS), supervised classification, suggestion extraction |
| `dashboard/` | Host-facing frontend (`index.html`) |
| `flask_api/` | Backend API that connects the frontend to the trained models |

Notebooks were developed and run in **Google Colab** with Google Drive mounted.

---

## Large Data Files — Google Drive

The raw data and intermediate matrices are too large for GitHub and are stored on Google Drive:

📁 **Google Drive:** `[https://drive.google.com/drive/folders/1sgpAYAjGLfWeoYnOZ_5XqWWhT5ynoLwH?usp=drive_link]`

Files on Drive: `raw_reviews.csv`, `MergedCleaned.csv`, `labelled_reviews.csv`, `tfidf_matrix.pkl`, `bow_matrix.pkl`, `w2v_embeddings.pkl`, `lda_topic_assignments.csv`, `task3_supervised_output.csv`, `sentence_aspect_polarity.csv`, `suggestions_by_aspect.csv`

---

## Running the Frontend

The dashboard is a standalone HTML file with two tabs — a live review analyser and Tableau district dashboards.

**Step 1 — Start the API**
```bash
cd flask_api
pip install -r requirements.txt
python app.py
```

**Step 2 — Open the dashboard**

Open `Actual Work/dashboard/index.html` in any browser. No build step required.

> The API must be running for the review analyser tab to return real model predictions. If you want to run the frontend without the API, open `index.html`, find `const API_ENDPOINT` near the top of the script, and set it to `null`.

---

## Key Results

### Task 1 — Sentiment Classification

| Model | Weighted F1 | Macro F1 |
|---|---|---|
| Logistic Regression (TF-IDF) | 0.9618 | 0.71 |
| RoBERTa (pretrained) | 0.9474 | 0.64 |

### Task 2 — Topic Modelling (BERTopic, 8 clusters)

| Topic | Count | Theme |
|---|---|---|
| 0 | 78,611 | General Bangkok stay experience |
| 1 | 64,942 | Cleanliness & property quality |
| 2 | 62,451 | Apartment facilities & amenities |
| 3 | 54,950 | Host responsiveness |
| 4 | 51,126 | Transport & BTS accessibility |
| 5 | 26,303 | Gratitude & positive experience |
| 6 | 23,033 | General positive stay |
| 7 | 19,089 | Airbnb host & apartment overall |

### Task 3 — Aspect Sentiment (Mean NSS across all listings)

| Aspect | Mean NSS | Interpretation |
|---|---|---|
| Communication | 0.8935 | Highest rated — hosts are responsive |
| Location | 0.8853 | Strong — proximity to BTS valued |
| Value | 0.8777 | Generally good perceived value |
| Cleanliness | 0.8517 | Good but bimodal — minority of listings drag score down |
| Amenities | 0.6368 | Most variable — room for improvement |
| Check-in | 0.6130 | Lowest & highest variance — inconsistent across listings |