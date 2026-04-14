import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


import os
import re
import pickle
import math
from collections import Counter
import nltk
import ftfy
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Download NLTK data (only needed on first run)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

# ── Preprocessing (inlined from preprocess.py — avoids gensim dependency) ────
# Mirrors preprocess.py clean_text() exactly so vectorizer output is consistent
STOPWORDS  = set(stopwords.words('english')) - {'not', 'no', 'never', 'very', 'but', 'however'}
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return '', []
    text = ftfy.fix_text(text)
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)         # remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # remove punctuation
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isascii()]
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) >= 2]
    return ' '.join(tokens), tokens


def to_match_tokens(text):
    if not isinstance(text, str):
        return set()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    return {t for t in text.split() if len(t) >= 3 and t not in STOPWORDS}

# ── Load models ───────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
ACTUAL_WORK_DIR = os.path.abspath(os.path.join(PROJECT_DIR, 'Actual Work'))
MODELS_DIR = os.path.abspath(os.path.join(ACTUAL_WORK_DIR, 'models'))
TASK3_DIR = os.path.abspath(os.path.join(ACTUAL_WORK_DIR, 'task3-combination'))
DASHBOARD_DIR = os.path.abspath(os.path.join(ACTUAL_WORK_DIR, 'dashboard'))
NSS_SCORES_PATH = os.path.abspath(os.path.join(TASK3_DIR, 'nss_scores.csv'))
IMPROVEMENT_SUGGESTIONS_PATH = os.path.abspath(os.path.join(TASK3_DIR, 'improvement_suggestions.csv'))
TASK2_TOPICS_PATH = os.path.abspath(os.path.join(ACTUAL_WORK_DIR, 'task2-topics', 'BERT_Gemini_Final_8.csv'))
LDA_TOPIC_ASSIGNMENTS_PATH = os.path.abspath(os.path.join(ACTUAL_WORK_DIR, 'task2-topics', 'lda_topic_assignments.csv'))

print(f"DEBUG: Looking for models in: {os.path.abspath(MODELS_DIR)}")
print(f"DEBUG: Dashboard directory: {DASHBOARD_DIR}")
print(f"DEBUG: Loading NSS scores from: {NSS_SCORES_PATH}")
print(f"DEBUG: Loading improvement suggestions from: {IMPROVEMENT_SUGGESTIONS_PATH}")
print(f"DEBUG: Loading topic summary from: {TASK2_TOPICS_PATH}")
print(f"DEBUG: Loading optional LDA assignments from: {LDA_TOPIC_ASSIGNMENTS_PATH}")

def load(filename):
    with open(os.path.join(MODELS_DIR, filename), 'rb') as f:
        return pickle.load(f)

print('Loading models...')
lr_sentiment_task1 = load('lr_sentiment_task1.pkl')
tfidf_task1        = load('tfidf_vectorizer_task1.pkl')
lr_aspect          = load('lr_aspect.pkl')
lr_sentiment_task3 = load('lr_sentiment_task3.pkl')
tfidf_task3        = load('tfidf_vectorizer_task3.pkl')
le_aspect          = load('le_aspect.pkl')
le_sentiment       = load('le_sentiment.pkl')
print('All models loaded.')

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

ASPECT_ORDER = ['cleanliness', 'location', 'communication', 'check_in', 'value', 'amenities']

DEFAULT_BASELINE_NSS = {
    'cleanliness': 0.8517,
    'location': 0.8853,
    'communication': 0.8935,
    'check_in': 0.6130,
    'value': 0.8777,
    'amenities': 0.6368,
}

# ── Aspect keyword dictionary (mirrors task3_rulebased.ipynb) ─────────────────
ASPECT_KEYWORDS = {
    'cleanliness':   ['clean','dirty','smell','dusty','dust','spotless','filthy','stain',
                      'stained','messy','odor','odour','stinky','mold','mouldy','gross',
                      'grimy','grime','reeked','smelly','unclean','damp','musty'],
    'location':      ['location','nearby','mrt','bts','skytrain','close','near','closeby',
                      'metro','subway','train','convenient','central','transport','commute',
                      'accessible','station'],
    'communication': ['host','responsive','reply','replied','response','communicate',
                      'helpful','attentive','friendly','accommodating','answer','answered',
                      'supportive','proactive','hostess','owner','prompt'],
    'check_in':      ['checkin','check-in','check in','checkout','check out','key','keycard',
                      'lockbox','passcode','arrival','departure','access','password','code',
                      'door','entry'],
    'value':         ['value','price','pricing','worth','money','cheap','expensive',
                      'affordable','overpriced','rate','cost','budget','steal','reasonable'],
    'amenities':     ['wifi','internet','kitchen','aircon','air conditioning','fridge','pool',
                      'gym','shower','towel','microwave','oven','washer','laundry','tv',
                      'bathroom','toilet','balcony','elevator','lift'],
}

# Signal phrases for suggestion extraction (mirrors task3_suggestion_extraction.ipynb)
SIGNAL_PHRASES = [
    'should','could','wish','needs','need to','missing','would be better',
    'hope','would have been','recommend','suggest','next time','improve',
    'fix','add','update','unfortunately','however','but','except','though',
    'would have liked','would prefer','would appreciate','lacking',
]

FALLBACK_ROADMAP = {
    'cleanliness': 'Deep-clean high-touch surfaces and refresh linens between each stay.',
    'location': 'Share a short transport guide with nearest BTS/MRT routes and travel time.',
    'communication': 'Commit to response SLAs and send proactive check-in/check-out updates.',
    'check_in': 'Simplify arrival with a clear lockbox/key handover guide and photo steps.',
    'value': 'Improve perceived value with transparent pricing and essential welcome items.',
    'amenities': 'Prioritise reliability checks for wifi, aircon, shower pressure, and kitchen basics.',
}

ROADMAP_PLAYBOOK = {
    'cleanliness': [
        'Deep-clean bathroom surfaces, linens, and odor-prone areas before each check-in.',
        'Use a pre-arrival photo checklist for kitchen, bathroom, and bedding hygiene.',
        'Escalate repeated cleanliness complaints to housekeeping within 24 hours.',
    ],
    'location': [
        'Share a transport quick-guide with BTS/MRT routes and expected travel times.',
        'Add noise guidance and quiet-hour expectations in pre-arrival messages.',
        'Offer nearby convenience and dining recommendations in one short map link.',
    ],
    'communication': [
        'Set a response SLA and confirm key guest questions within one hour.',
        'Send proactive updates for check-in, Wi-Fi, and support contacts.',
        'Use a standardized message template for arrival and issue resolution.',
    ],
    'check_in': [
        'Rewrite check-in steps with clear numbered instructions and photos.',
        'Validate lockbox code and entry instructions before every arrival.',
        'Add a same-day arrival message with backup contact and map pin.',
    ],
    'value': [
        'Align pricing with unit condition and highlight included essentials clearly.',
        'Bundle small extras to increase perceived value at current price points.',
        'Review competitor pricing weekly and adjust for demand and seasonality.',
    ],
    'amenities': [
        'Test Wi-Fi speed, AC cooling, and hot-water reliability before each stay.',
        'Restock towels and toiletries using a fixed turnover checklist.',
        'Prioritize repairs for amenities that appear in repeated complaints.',
    ],
}

LDA_TOPIC_LABELS = {
    0: 'Neighbourhood & Dining',
    1: 'Check-in & Arrival',
    2: 'General Positive Sentiment',
    3: 'Overall Experience',
    4: 'Facilities & Amenities',
    5: 'Transport & Accessibility',
    6: 'Complaints & Issues',
}

ISSUE_ORIENTED_TOPICS = {
    'Complaints & Issues',
    'Check-in & Arrival',
    'Facilities & Amenities',
}

WORD_CLOUD_NOISE_TERMS = {
    # Explicitly remove low-signal connector words.
    'not', 'but', 'and', 'or', 'if', 'then', 'than', 'also', 'even', 'just', 'still',
    'really', 'very', 'much', 'many', 'more', 'most', 'some', 'any',
    'there', 'here', 'where', 'when', 'while', 'because', 'though', 'however',
    # Common conversational contractions / artifacts.
    'dont', 'didnt', 'cant', 'wont', 'isnt', 'wasnt', 'werent', 'couldnt', 'wouldnt', 'shouldnt',
    'im', 'ive', 'youre', 'weve', 'theyre', 'thats', 'theres', 'its',
}

WORD_CLOUD_STOPWORDS = STOPWORDS.union(WORD_CLOUD_NOISE_TERMS).union({
    'airbnb', 'bangkok', 'place', 'stay', 'host', 'apartment', 'room', 'property',
    'would', 'could', 'should', 'also', 'really', 'very', 'still', 'much', 'many',
    'one', 'two', 'got', 'get', 'like', 'overall', 'though', 'however', 'thing',
    'time', 'day', 'night', 'next', 'back', 'again', 'staycation', 'nice', 'good',
})

# Send enough terms so the frontend top-words control (up to 80) can be honored.
WORD_CLOUD_MAX_TERMS = 120

# Keep enough polarity tokens per side so the friction filter can show up to 40 total terms.
FRICTION_MAX_TERMS_PER_SIDE = 30

ASPECT_DISPLAY = {
    'cleanliness': 'Cleanliness',
    'location': 'Location',
    'communication': 'Communication',
    'check_in': 'Check-in',
    'value': 'Value for Money',
    'amenities': 'Amenities',
}

ACTION_CATEGORY_KEYWORDS = {
    'Internet Reliability': ['wifi', 'internet', 'connection', 'network', 'signal', 'slow', 'speed', 'lag'],
    'Bathroom Comfort': ['bathroom', 'shower', 'toilet', 'water pressure', 'hot water', 'drain', 'sink'],
    'Supply Shortage: Towels': ['towel', 'toiletries', 'soap', 'shampoo', 'blanket', 'pillow'],
    'Check-in Friction': ['check in', 'check-in', 'arrival', 'key', 'lockbox', 'passcode', 'code', 'entry', 'door'],
    'Noise Control': ['noise', 'noisy', 'loud', 'construction', 'traffic', 'street noise'],
    'Air Conditioning': ['aircon', 'air conditioning', 'ac', 'cooling', 'temperature', 'hot'],
    'Kitchen Setup': ['kitchen', 'cook', 'utensil', 'stove', 'microwave', 'fridge', 'kettle'],
    'Host Responsiveness': ['host', 'reply', 'responsive', 'response', 'communication', 'message'],
    'Value for Money': ['price', 'value', 'cost', 'overpriced', 'expensive', 'worth'],
    'Cleanliness Standard': ['clean', 'dirty', 'smell', 'dust', 'mold', 'stain'],
}

CATEGORY_TO_ASPECT = {
    'Internet Reliability': 'amenities',
    'Bathroom Comfort': 'amenities',
    'Supply Shortage: Towels': 'amenities',
    'Check-in Friction': 'check_in',
    'Noise Control': 'location',
    'Air Conditioning': 'amenities',
    'Kitchen Setup': 'amenities',
    'Host Responsiveness': 'communication',
    'Value for Money': 'value',
    'Cleanliness Standard': 'cleanliness',
}

# Stricter business thresholds for issue-rate benchmarking.
BENCHMARK_GREEN_DELTA = -0.02  # At least 2pp below Bangkok benchmark.
BENCHMARK_RED_DELTA = 0.04     # More than 4pp above benchmark.

# Keep impact estimates conservative and decision-friendly.
IMPACT_MIN_PCT = 4
IMPACT_MAX_PCT = 22


def classify_benchmark_status(issue_rate, benchmark):
    if benchmark is None:
        return 'amber'

    delta = float(issue_rate) - float(benchmark)
    if delta <= BENCHMARK_GREEN_DELTA:
        return 'green'
    if delta > BENCHMARK_RED_DELTA:
        return 'red'
    return 'amber'


def estimate_potential_impact_pct(mention_share, aspect_gap, issue_rate, benchmark_delta):
    # Weighted score: category concentration + aspect performance gap + issue pressure.
    raw_score = (
        (float(mention_share) * 24)
        + (float(aspect_gap) * 36)
        + (max(0.0, float(benchmark_delta)) * 50)
        + (float(issue_rate) * 8)
    )
    bounded = max(IMPACT_MIN_PCT, min(IMPACT_MAX_PCT, raw_score))
    return int(round(bounded))


def nss_label(nss_value):
    return (
        'Excellent' if nss_value >= 0.7 else
        'Good' if nss_value >= 0.3 else
        'Mixed' if nss_value >= -0.3 else
        'Poor' if nss_value >= -0.7 else
        'Critical'
    )


def load_baseline_nss(path):
    try:
        df = pd.read_csv(path)
        if 'aspect' not in df.columns or 'NSS' not in df.columns:
            raise ValueError('nss_scores.csv missing required columns: aspect, NSS')
        means = df.groupby('aspect', dropna=True)['NSS'].mean().to_dict()
        baseline = {aspect: round(float(means.get(aspect, DEFAULT_BASELINE_NSS[aspect])), 4)
                    for aspect in ASPECT_ORDER}
        return baseline
    except Exception as exc:
        print(f'WARNING: Failed to load NSS baselines, using defaults. Error: {exc}')
        return DEFAULT_BASELINE_NSS.copy()


def infer_aspects_from_text(text):
    lowered = text.lower()
    return [
        aspect for aspect, keywords in ASPECT_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]


def build_suggestion_library(path, max_per_aspect=450):
    library = {aspect: [] for aspect in ASPECT_ORDER}
    seen = {aspect: set() for aspect in ASPECT_ORDER}

    if not os.path.exists(path):
        print(f'WARNING: Suggestion file not found: {path}')
        return library

    try:
        df = pd.read_csv(path)
        if 'comments' not in df.columns:
            raise ValueError('improvement_suggestions.csv missing required column: comments')

        for comment in df['comments'].dropna().astype(str):
            for sentence in sent_tokenize(comment):
                sentence = ' '.join(sentence.split())
                if len(sentence) < 20:
                    continue
                sentence_lower = sentence.lower()
                if not any(phrase in sentence_lower for phrase in SIGNAL_PHRASES):
                    continue

                matched_aspects = infer_aspects_from_text(sentence_lower)
                if not matched_aspects:
                    continue

                candidate_tokens = to_match_tokens(sentence)
                if not candidate_tokens:
                    continue

                signature = sentence_lower[:260]
                for aspect in matched_aspects:
                    if len(library[aspect]) >= max_per_aspect:
                        continue
                    if signature in seen[aspect]:
                        continue
                    library[aspect].append({'text': sentence, 'tokens': candidate_tokens})
                    seen[aspect].add(signature)

    except Exception as exc:
        print(f'WARNING: Failed to build suggestion library. Error: {exc}')

    return library


def concise_roadmap_action(text, aspect, max_words=14):
    cleaned = ' '.join(str(text).replace('\n', ' ').split())
    cleaned = cleaned.replace('•', ' ').replace('–', '-').strip(' -')

    if not cleaned:
        return FALLBACK_ROADMAP[aspect]

    aspect_keywords = ASPECT_KEYWORDS.get(aspect, [])
    clauses = [
        c.strip(' -.,:;')
        for c in re.split(r'[.!?;]|\s+-\s+', cleaned)
        if c and c.strip()
    ]

    preferred = [
        c for c in clauses
        if any(k in c.lower() for k in aspect_keywords)
    ]
    pool = preferred if preferred else clauses

    chosen = ''
    for clause in pool:
        words = clause.split()
        if 4 <= len(words) <= 22:
            chosen = clause
            break
    if not chosen:
        chosen = pool[0] if pool else cleaned

    words = chosen.split()
    if len(words) > max_words:
        chosen = ' '.join(words[:max_words])

    chosen = chosen.strip(' -.,:;')
    if not chosen:
        return FALLBACK_ROADMAP[aspect]

    return chosen[0].upper() + chosen[1:]


def sanitize_for_json(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    return value


BASELINE_NSS = load_baseline_nss(NSS_SCORES_PATH)
SUGGESTION_LIBRARY = build_suggestion_library(IMPROVEMENT_SUGGESTIONS_PATH)


def load_dashboard_data():
    payload = {
        'sentiment_by_neighbourhood': [],
        'aspect_scores_by_neighbourhood': [],
        'topic_distribution_lda': [],
        'topic_distribution_by_neighbourhood': [],
        'improvement_signals_by_neighbourhood': [],
        'action_oriented_board_by_neighbourhood': [],
        'issue_oriented_suggestions_by_neighbourhood': [],
        'improvement_wordcloud_by_neighbourhood': [],
        'friction_terms_by_neighbourhood': [],
        'district_overview': [],
        'issue_rate_benchmark': None,
        'executive_summary': '',
        'executive_summary_by_neighbourhood': {},
        'selected_neighbourhood_default': None,
        'notes': [],
    }

    def tokenize_for_wordcloud(text):
        tokens = re.findall(r"[a-zA-Z][a-zA-Z\-']+", str(text).lower())
        cleaned = []
        for t in tokens:
            normalized = t.replace("'", '')
            if len(normalized) < 3:
                continue
            if normalized.isdigit():
                continue
            if t in WORD_CLOUD_STOPWORDS or normalized in WORD_CLOUD_STOPWORDS:
                continue
            cleaned.append(normalized)
        return cleaned

    def infer_action_categories(text):
        lowered = str(text).lower()
        cats = []
        for category, keywords in ACTION_CATEGORY_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                cats.append(category)
        return cats

    try:
        nss_df = pd.read_csv(NSS_SCORES_PATH)
        required_nss_cols = {'listing_id', 'neighbourhood', 'aspect', 'NA', 'PA', 'NSS'}
        if not required_nss_cols.issubset(nss_df.columns):
            missing = required_nss_cols.difference(set(nss_df.columns))
            raise ValueError(f'nss_scores.csv missing columns: {sorted(missing)}')

        nss_df = nss_df.dropna(subset=['neighbourhood'])
        nss_df['NA'] = nss_df['NA'].fillna(0)
        nss_df['PA'] = nss_df['PA'].fillna(0)
        nss_df['NSS'] = nss_df['NSS'].fillna(0.0)

        aspect_benchmark = nss_df.groupby('aspect', dropna=True)['NSS'].mean().to_dict()

        # Better sentiment view: include rates, net sentiment, and volume.
        sent = (
            nss_df.groupby('neighbourhood', as_index=False)
            .agg(positive=('PA', 'sum'), negative=('NA', 'sum'))
        )
        sent['positive'] = sent['positive'].astype(int)
        sent['negative'] = sent['negative'].astype(int)
        sent['total'] = (sent['positive'] + sent['negative']).astype(int)
        sent = sent[sent['total'] > 0]
        sent['positive_rate'] = (sent['positive'] / sent['total']).round(4)
        sent['negative_rate'] = (sent['negative'] / sent['total']).round(4)
        sent['net_sentiment'] = ((sent['positive'] - sent['negative']) / sent['total']).round(4)
        sent = sent.sort_values(['negative_rate', 'total'], ascending=[False, False]).head(18)
        payload['sentiment_by_neighbourhood'] = sent.to_dict(orient='records')

        # Aspect heatmap rows.
        aspect_pivot = (
            nss_df.pivot_table(index='neighbourhood', columns='aspect', values='NSS', aggfunc='mean')
            .reset_index()
        )
        for aspect in ASPECT_ORDER:
            if aspect not in aspect_pivot.columns:
                aspect_pivot[aspect] = 0.0

        volume = (
            nss_df.groupby('neighbourhood', as_index=False)
            .size()
            .rename(columns={'size': 'records'})
        )
        aspect_pivot = aspect_pivot.merge(volume, on='neighbourhood', how='left')
        aspect_pivot['avg_nss'] = aspect_pivot[ASPECT_ORDER].mean(axis=1)
        aspect_pivot = aspect_pivot.sort_values(['avg_nss', 'records'], ascending=[True, False]).head(18)

        aspect_rows = []
        for _, row in aspect_pivot.iterrows():
            aspect_rows.append({
                'neighbourhood': row['neighbourhood'],
                'cleanliness': round(float(row.get('cleanliness', 0.0)), 4),
                'location': round(float(row.get('location', 0.0)), 4),
                'communication': round(float(row.get('communication', 0.0)), 4),
                'check_in': round(float(row.get('check_in', 0.0)), 4),
                'value': round(float(row.get('value', 0.0)), 4),
                'amenities': round(float(row.get('amenities', 0.0)), 4),
                'avg_nss': round(float(row.get('avg_nss', 0.0)), 4),
                'records': int(row.get('records', 0)),
            })
        payload['aspect_scores_by_neighbourhood'] = aspect_rows

        listing_map = (
            nss_df[['listing_id', 'neighbourhood']]
            .dropna(subset=['listing_id', 'neighbourhood'])
            .drop_duplicates(subset=['listing_id'])
        )

        suggestions_df = pd.DataFrame()
        merged = pd.DataFrame()
        issue_source = pd.DataFrame()
        signals = pd.DataFrame()
        action_by_neighbourhood = {}
        global_action_counter = Counter()

        if os.path.exists(IMPROVEMENT_SUGGESTIONS_PATH):
            suggestions_df = pd.read_csv(IMPROVEMENT_SUGGESTIONS_PATH)
            if {'listing_id', 'predicted_aspect', 'comments'}.issubset(suggestions_df.columns):
                merged = suggestions_df.merge(listing_map, on='listing_id', how='left')
                merged = merged.dropna(subset=['neighbourhood'])
                merged['predicted_aspect'] = merged['predicted_aspect'].astype(str)
                merged['comments'] = merged['comments'].fillna('').astype(str)
                merged['is_issue'] = merged['predicted_aspect'].isin(ISSUE_ORIENTED_TOPICS)

        if not merged.empty:
            issue_rate_benchmark = float(merged['is_issue'].mean())
            payload['issue_rate_benchmark'] = round(issue_rate_benchmark, 4)

            signals = (
                merged.groupby('neighbourhood', as_index=False)
                .agg(
                    total_suggestions=('listing_id', 'size'),
                    issue_suggestions=('is_issue', 'sum'),
                )
            )
            signals['issue_rate'] = (signals['issue_suggestions'] / signals['total_suggestions']).round(4)
            signals['benchmark_issue_rate'] = round(issue_rate_benchmark, 4)
            signals['benchmark_delta'] = (signals['issue_rate'] - issue_rate_benchmark).round(4)
            signals['benchmark_status'] = signals['issue_rate'].apply(
                lambda x: classify_benchmark_status(float(x), issue_rate_benchmark)
            )
            signals['issue_rate_pct'] = (signals['issue_rate'] * 100).round(2)

            top_aspect = (
                merged.groupby(['neighbourhood', 'predicted_aspect'], as_index=False)
                .size()
                .sort_values(['neighbourhood', 'size'], ascending=[True, False])
                .drop_duplicates(subset=['neighbourhood'])
                .rename(columns={'predicted_aspect': 'dominant_aspect', 'size': 'dominant_aspect_count'})
            )

            signals = signals.merge(
                top_aspect[['neighbourhood', 'dominant_aspect', 'dominant_aspect_count']],
                on='neighbourhood',
                how='left'
            )
            signals = signals.sort_values(['issue_rate', 'issue_suggestions'], ascending=[False, False]).head(18)
            payload['improvement_signals_by_neighbourhood'] = signals.to_dict(orient='records')

            # Action-oriented board with key action extraction + impact estimate.
            issue_rows = []
            issue_source = merged[merged['is_issue']]
            issue_grouped = (
                issue_source.groupby('neighbourhood', as_index=False)
                .agg(issue_suggestions=('listing_id', 'size'))
                .sort_values('issue_suggestions', ascending=False)
                .head(12)
            )

            signal_map = {
                r['neighbourhood']: r
                for r in payload['improvement_signals_by_neighbourhood']
            }
            aspect_map = {r['neighbourhood']: r for r in payload['aspect_scores_by_neighbourhood']}

            for _, row in issue_grouped.iterrows():
                nbh = row['neighbourhood']
                subset = issue_source[issue_source['neighbourhood'] == nbh]
                dominant_issue = (
                    subset['predicted_aspect'].value_counts().idxmax()
                    if not subset.empty else 'Complaints & Issues'
                )

                category_counter = Counter()
                snippets = []
                seen_snippets = set()

                for comment in subset['comments'].head(1000):
                    for sentence in sent_tokenize(comment):
                        s = ' '.join(sentence.split())
                        if len(s) < 25:
                            continue
                        lowered = s.lower()
                        cats = infer_action_categories(lowered)
                        for cat in cats:
                            category_counter[cat] += 1
                        if any(phrase in lowered for phrase in SIGNAL_PHRASES):
                            if s not in seen_snippets and len(snippets) < 3:
                                snippets.append(s)
                                seen_snippets.add(s)

                if not category_counter:
                    fallback_cat = 'Check-in Friction' if dominant_issue == 'Check-in & Arrival' else 'Value for Money'
                    category_counter[fallback_cat] = int(max(1, row['issue_suggestions']))

                total_mentions = max(1, sum(category_counter.values()))
                district_signal = signal_map.get(nbh, {})
                district_issue_rate = float(district_signal.get('issue_rate', issue_rate_benchmark))
                district_benchmark_status = district_signal.get('benchmark_status', 'amber')
                district_aspects = aspect_map.get(nbh, {})

                categories = []
                for cat, mentions in category_counter.most_common(5):
                    mention_share = mentions / total_mentions
                    impacted_aspect = CATEGORY_TO_ASPECT.get(cat, 'value')
                    district_aspect_score = float(district_aspects.get(impacted_aspect, 0.0))
                    benchmark_aspect_score = float(aspect_benchmark.get(impacted_aspect, 0.0))
                    aspect_gap = max(0.0, benchmark_aspect_score - district_aspect_score)
                    benchmark_delta = max(0.0, district_issue_rate - issue_rate_benchmark)

                    potential_impact_pct = estimate_potential_impact_pct(
                        mention_share=mention_share,
                        aspect_gap=aspect_gap,
                        issue_rate=district_issue_rate,
                        benchmark_delta=benchmark_delta,
                    )

                    categories.append({
                        'category': cat,
                        'mentions': int(mentions),
                        'mention_share': round(float(mention_share), 4),
                        'frequency_tag': f'{cat} (Mentioned {int(mentions)} times)',
                        'impacted_aspect': impacted_aspect,
                        'impacted_aspect_label': ASPECT_DISPLAY.get(impacted_aspect, impacted_aspect),
                        'potential_impact_pct': potential_impact_pct,
                        'potential_impact_text': (
                            f"If fixed, {ASPECT_DISPLAY.get(impacted_aspect, impacted_aspect)} is predicted to rise by ~{potential_impact_pct}%"
                        ),
                    })

                for cat, mentions in category_counter.items():
                    global_action_counter[cat] += mentions

                action_row = {
                    'neighbourhood': nbh,
                    'dominant_issue': dominant_issue,
                    'issue_suggestions': int(row['issue_suggestions']),
                    'issue_rate': round(district_issue_rate, 4),
                    'benchmark_status': district_benchmark_status,
                    'categories': categories,
                    'sample_suggestions': snippets,
                }

                issue_rows.append(action_row)
                action_by_neighbourhood[nbh] = action_row

            payload['action_oriented_board_by_neighbourhood'] = issue_rows
            payload['issue_oriented_suggestions_by_neighbourhood'] = issue_rows
            if issue_rows:
                payload['selected_neighbourhood_default'] = issue_rows[0]['neighbourhood']

            # Word cloud terms per district from issue-oriented feedback.
            wordcloud_rows = []
            for _, row in issue_grouped.iterrows():
                nbh = row['neighbourhood']
                subset = issue_source[issue_source['neighbourhood'] == nbh]
                counter = Counter()
                for comment in subset['comments'].head(2000):
                    counter.update(tokenize_for_wordcloud(comment))

                words = [
                    {'word': word, 'weight': int(weight)}
                    for word, weight in counter.most_common(WORD_CLOUD_MAX_TERMS)
                ]
                wordcloud_rows.append({
                    'neighbourhood': nbh,
                    'words': words,
                })

            payload['improvement_wordcloud_by_neighbourhood'] = wordcloud_rows

            # Topic distribution by neighbourhood (LDA-aligned labels from Task 3 pipeline).
            topic_by_nbh = (
                merged.groupby(['neighbourhood', 'predicted_aspect'], as_index=False)
                .size()
                .rename(columns={'predicted_aspect': 'topic_label', 'size': 'count'})
            )
            topic_by_nbh = topic_by_nbh.sort_values(['neighbourhood', 'count'], ascending=[True, False])
            payload['topic_distribution_by_neighbourhood'] = topic_by_nbh.head(180).to_dict(orient='records')

            # LDA topic distribution overall.
            topic_overall = (
                merged.groupby('predicted_aspect', as_index=False)
                .size()
                .rename(columns={'predicted_aspect': 'topic_label', 'size': 'count'})
                .sort_values('count', ascending=False)
            )
            payload['topic_distribution_lda'] = topic_overall.to_dict(orient='records')
            payload['notes'].append('LDA topic chart uses LDA-aligned labels from improvement_suggestions.csv.')
            payload['notes'].append(
                'Benchmark status thresholds: green <= -2pp vs Bangkok benchmark, amber between -2pp and +4pp, red > +4pp.'
            )

            # Friction terms: top positive/negative drivers by district.
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                analyzer = SentimentIntensityAnalyzer()
                friction_rows = []

                districts_for_friction = [
                    r['neighbourhood']
                    for r in payload['improvement_signals_by_neighbourhood'][:12]
                ]

                def build_term_splits(comments, max_comments=1200):
                    positive = Counter()
                    negative = Counter()
                    for comment in comments[:max_comments]:
                        for sentence in sent_tokenize(str(comment)):
                            sentence = sentence.strip()
                            if not sentence:
                                continue
                            score = analyzer.polarity_scores(sentence)['compound']
                            tokens = tokenize_for_wordcloud(sentence)
                            if not tokens:
                                continue
                            if score >= 0.05:
                                positive.update(tokens)
                            elif score <= -0.05:
                                negative.update(tokens)

                    # Remove overlapping tokens so a term does not appear as both a top positive
                    # and a top negative driver in the same view.
                    for token in set(positive).intersection(negative):
                        if positive[token] >= negative[token]:
                            negative.pop(token, None)
                        else:
                            positive.pop(token, None)

                    return {
                        'positive_terms': [
                            {'word': word, 'count': int(count)}
                            for word, count in positive.most_common(FRICTION_MAX_TERMS_PER_SIDE)
                        ],
                        'negative_terms': [
                            {'word': word, 'count': int(count)}
                            for word, count in negative.most_common(FRICTION_MAX_TERMS_PER_SIDE)
                        ],
                    }

                for nbh in districts_for_friction:
                    comments = merged.loc[merged['neighbourhood'] == nbh, 'comments'].astype(str).tolist()
                    splits = build_term_splits(comments)
                    friction_rows.append({'neighbourhood': nbh, **splits})

                overall_splits = build_term_splits(merged['comments'].astype(str).tolist(), max_comments=4000)
                friction_rows.append({'neighbourhood': '__overall__', **overall_splits})
                payload['friction_terms_by_neighbourhood'] = friction_rows
            except Exception as exc:
                payload['notes'].append(f'Could not compute friction terms: {exc}')

        # Optional direct LDA assignment file (preferred when available).
        if os.path.exists(LDA_TOPIC_ASSIGNMENTS_PATH):
            try:
                lda_df = pd.read_csv(LDA_TOPIC_ASSIGNMENTS_PATH)
                if {'listing_id', 'lda_topic_label'}.issubset(lda_df.columns):
                    if 'neighbourhood' in lda_df.columns:
                        lda_df = lda_df.dropna(subset=['neighbourhood'])
                    else:
                        lda_df = lda_df.merge(listing_map, on='listing_id', how='left').dropna(subset=['neighbourhood'])
                    lda_topic_overall = (
                        lda_df.groupby('lda_topic_label', as_index=False)
                        .size()
                        .rename(columns={'lda_topic_label': 'topic_label', 'size': 'count'})
                        .sort_values('count', ascending=False)
                    )
                    payload['topic_distribution_lda'] = lda_topic_overall.to_dict(orient='records')
                    payload['notes'].append('LDA topic chart uses lda_topic_assignments.csv (direct model assignments).')
            except Exception as exc:
                # Keep backend diagnostics in logs, but do not leak raw exceptions into UI notes.
                print(f'WARNING: Could not use lda_topic_assignments.csv. Error: {exc}')

        # Build district opportunity matrix rows.
        sent_overview = sent[['neighbourhood', 'positive_rate', 'negative_rate', 'net_sentiment', 'total']].copy()
        aspect_overview = pd.DataFrame(payload['aspect_scores_by_neighbourhood'])
        signal_overview = pd.DataFrame(payload['improvement_signals_by_neighbourhood'])

        if not sent_overview.empty and not aspect_overview.empty:
            summary = sent_overview.merge(
                aspect_overview[['neighbourhood', 'avg_nss', 'records']],
                on='neighbourhood',
                how='left'
            )
            if not signal_overview.empty:
                summary = summary.merge(
                    signal_overview[['neighbourhood', 'issue_rate', 'issue_suggestions', 'total_suggestions']],
                    on='neighbourhood',
                    how='left'
                )

            summary['issue_rate'] = summary['issue_rate'].fillna(0.0)
            summary['issue_suggestions'] = summary['issue_suggestions'].fillna(0).astype(int)
            summary['total_suggestions'] = summary['total_suggestions'].fillna(0).astype(int)
            summary['avg_nss'] = summary['avg_nss'].fillna(0.0)
            if payload['issue_rate_benchmark'] is not None:
                summary['benchmark_issue_rate'] = float(payload['issue_rate_benchmark'])
                summary['benchmark_status'] = summary['issue_rate'].apply(
                    lambda x: classify_benchmark_status(float(x), float(payload['issue_rate_benchmark']))
                )
            summary = summary.sort_values(['issue_rate', 'avg_nss'], ascending=[False, True]).head(18)
            payload['district_overview'] = summary.to_dict(orient='records')

        # Executive summary (overall + district level TLDR).
        if aspect_benchmark:
            sorted_aspects = sorted(aspect_benchmark.items(), key=lambda kv: kv[1])
            weakest_aspect, _ = sorted_aspects[0]
            strongest_aspect, _ = sorted_aspects[-1]
            top_action = global_action_counter.most_common(1)[0][0] if global_action_counter else 'Operational consistency'

            payload['executive_summary'] = (
                f"Your portfolio is over-performing in {ASPECT_DISPLAY.get(strongest_aspect, strongest_aspect)} "
                f"but requires immediate attention to {ASPECT_DISPLAY.get(weakest_aspect, weakest_aspect)} "
                f"({top_action}) to improve guest retention."
            )

            district_summaries = {}
            for row in payload['district_overview']:
                nbh = row['neighbourhood']
                nbh_df = nss_df[nss_df['neighbourhood'] == nbh]
                if nbh_df.empty:
                    continue
                nbh_aspects = nbh_df.groupby('aspect', dropna=True)['NSS'].mean().to_dict()
                if not nbh_aspects:
                    continue
                local_sorted = sorted(nbh_aspects.items(), key=lambda kv: kv[1])
                local_weakest, _ = local_sorted[0]
                local_strongest, _ = local_sorted[-1]
                top_local_action = (
                    action_by_neighbourhood.get(nbh, {}).get('categories', [{}])[0].get('category', 'Operational consistency')
                )
                status = row.get('benchmark_status', 'amber')
                urgency = 'immediate attention' if status == 'red' else ('targeted improvement' if status == 'amber' else 'light optimization')

                district_summaries[nbh] = (
                    f"{nbh} is strongest in {ASPECT_DISPLAY.get(local_strongest, local_strongest)} "
                    f"but needs {urgency} in {ASPECT_DISPLAY.get(local_weakest, local_weakest)} "
                    f"with focus on {top_local_action}."
                )

            payload['executive_summary_by_neighbourhood'] = district_summaries

        if not payload['selected_neighbourhood_default'] and payload['district_overview']:
            payload['selected_neighbourhood_default'] = payload['district_overview'][0]['neighbourhood']

        # Fallback to BERT file only when no LDA-aligned distribution is available.
        if not payload['topic_distribution_lda'] and os.path.exists(TASK2_TOPICS_PATH):
            topic_df = pd.read_csv(TASK2_TOPICS_PATH)
            if {'Topic', 'Count', 'Name'}.issubset(topic_df.columns):
                topic_df = topic_df.sort_values('Count', ascending=False)
                payload['topic_distribution_lda'] = [
                    {
                        'topic_label': str(row['Name']),
                        'count': int(row['Count']),
                    }
                    for _, row in topic_df.iterrows()
                ]
                payload['notes'].append('Fell back to BERTopic counts because LDA assignment file is not available locally.')

        if not payload['executive_summary']:
            payload['executive_summary'] = (
                'District insights loaded. Communication is generally strong, while targeted amenities fixes can improve retention.'
            )

    except Exception as exc:
        payload['notes'].append(f'Dashboard aggregation fallback used due to error: {exc}')
        print(f'WARNING: Dashboard aggregation failed. Error: {exc}')

    return payload


DASHBOARD_DATA = sanitize_for_json(load_dashboard_data())

# ── NSS computation (mirrors task3_rulebased.ipynb) ───────────────────────────
def compute_nss(raw_text):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer  = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(raw_text)
    counts    = {}

    for sentence in sentences:
        score = analyzer.polarity_scores(sentence)['compound']
        if score >= 0.05:
            polarity = 'positive'
        elif score <= -0.05:
            polarity = 'negative'
        else:
            continue  # neutral excluded from NSS

        lower = sentence.lower()
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                if aspect not in counts:
                    counts[aspect] = {'pos': 0, 'neg': 0}
                counts[aspect]['pos' if polarity == 'positive' else 'neg'] += 1

    results = {}
    for aspect, c in counts.items():
        total = c['pos'] + c['neg']
        if total == 0:
            continue
        nss   = round((c['pos'] - c['neg']) / total, 2)
        label = ('Excellent' if nss >= 0.7 else
                 'Good'      if nss >= 0.3 else
                 'Mixed'     if nss >= -0.3 else
                 'Poor'      if nss >= -0.7 else
                 'Critical')
        results[aspect] = {'nss': nss, 'label': label, 'pos': c['pos'], 'neg': c['neg']}

    return results

# ── Suggestion extraction (mirrors task3_suggestion_extraction.ipynb) ─────────
def extract_suggestions(raw_text):
    sentences = sent_tokenize(raw_text)
    return [
        s.strip() for s in sentences
        if any(p in s.lower() for p in SIGNAL_PHRASES) and len(s.strip()) > 20
    ][:4]


def complete_aspect_scores(review_aspects):
    completed = {}
    for aspect in ASPECT_ORDER:
        if aspect in review_aspects:
            nss = float(review_aspects[aspect]['nss'])
            completed[aspect] = {
                'nss': round(nss, 2),
                'label': nss_label(nss),
                'source': 'review',
                'pos': int(review_aspects[aspect]['pos']),
                'neg': int(review_aspects[aspect]['neg']),
            }
        else:
            nss = float(BASELINE_NSS.get(aspect, 0.0))
            completed[aspect] = {
                'nss': round(nss, 2),
                'label': nss_label(nss),
                'source': 'baseline',
            }
    return completed


def contextual_roadmap_action(raw_text, aspect):
    lowered = str(raw_text).lower()

    if aspect == 'amenities':
        if any(k in lowered for k in ['wifi', 'internet', 'signal', 'speed']):
            return 'Stabilize Wi-Fi and verify connection speed before each check-in.'
        if any(k in lowered for k in ['aircon', 'air conditioning', 'ac', 'cooling']):
            return 'Service AC units and confirm cooling performance before guest arrival.'
        if any(k in lowered for k in ['shower', 'water pressure', 'hot water', 'bathroom']):
            return 'Fix bathroom reliability issues and test water pressure daily.'

    if aspect == 'check_in':
        if any(k in lowered for k in ['check in', 'check-in', 'arrival', 'key', 'lockbox', 'code', 'entry']):
            return 'Simplify check-in with photo steps, code verification, and a same-day arrival message.'

    if aspect == 'cleanliness':
        if any(k in lowered for k in ['dirty', 'smell', 'dust', 'mold', 'stain', 'clean']):
            return 'Close cleanliness gaps with a bathroom-odor and linens deep-clean checklist.'

    if aspect == 'communication':
        if any(k in lowered for k in ['response', 'responsive', 'reply', 'message', 'host']):
            return 'Improve guest communication with faster response SLAs and proactive status updates.'

    if aspect == 'location':
        if any(k in lowered for k in ['noise', 'traffic', 'transport', 'bts', 'mrt', 'location']):
            return 'Set clear location expectations and share transport and noise guidance pre-arrival.'

    if aspect == 'value':
        if any(k in lowered for k in ['price', 'value', 'cost', 'overpriced', 'expensive', 'worth']):
            return 'Improve value perception through clearer inclusions and competitive pricing checks.'

    return None


def build_roadmap_suggestions(raw_text, completed_aspects, review_suggestions, threshold=0.2):
    roadmap = {}

    for aspect in ASPECT_ORDER:
        nss = float(completed_aspects[aspect]['nss'])
        if nss >= threshold:
            continue

        selected = []
        contextual = contextual_roadmap_action(raw_text, aspect)
        if contextual:
            selected.append(contextual)

        for playbook_action in ROADMAP_PLAYBOOK.get(aspect, []):
            if len(selected) >= 3:
                break
            if playbook_action not in selected:
                selected.append(playbook_action)

        if not selected:
            selected = [FALLBACK_ROADMAP[aspect]]

        roadmap[aspect] = selected[:3]

    return roadmap

# ── API endpoint ──────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def dashboard_home():
    index_path = os.path.join(DASHBOARD_DIR, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(DASHBOARD_DIR, 'index.html')
    return jsonify({
        'error': 'Dashboard file not found',
        'expected_path': index_path,
    }), 404


@app.route('/dashboard', methods=['GET'])
def dashboard_alias():
    return dashboard_home()


@app.route('/dashboard-data', methods=['GET'])
def dashboard_data():
    return jsonify(sanitize_for_json(DASHBOARD_DATA))


@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    if request.method == 'GET':
        return jsonify({
            'message': 'Use POST /analyse with JSON body: {"review": "your review text"}',
            'example': {
                'url': '/analyse',
                'method': 'POST',
                'headers': {'Content-Type': 'application/json'},
                'body': {'review': 'Great location near BTS, but wifi was slow.'},
            },
            'dashboard': '/',
            'health': '/health',
        })

    data = request.get_json()
    if not data or 'review' not in data:
        return jsonify({'error': 'Missing "review" field'}), 400

    raw_text = data['review'].strip()
    if len(raw_text) < 15:
        return jsonify({'error': 'Review too short (minimum 15 characters)'}), 400

    cleaned_text, _ = clean_text(raw_text)

    # Task 1: overall sentiment via LR trained on full TF-IDF corpus
    vec1           = tfidf_task1.transform([cleaned_text])
    sentiment_pred = lr_sentiment_task1.predict(vec1)[0]
    confidence     = round(float(lr_sentiment_task1.predict_proba(vec1)[0].max()) * 100)

    # Task 2: topic/aspect category via LR trained on LDA-labelled data
    vec3        = tfidf_task3.transform([cleaned_text])
    aspect_pred = le_aspect.inverse_transform(lr_aspect.predict(vec3))[0]

    # Task 3: aspect NSS scores (rule-based, real VADER + full 6-aspect completion)
    review_aspects = compute_nss(raw_text)
    aspects = complete_aspect_scores(review_aspects)

    # Task 3: improvement signals from current review + historical keyword-matched library
    suggestions = extract_suggestions(raw_text)
    roadmap = build_roadmap_suggestions(raw_text, aspects, suggestions, threshold=0.2)

    # Keep flat suggestions for backwards compatibility while exposing aspect-wise roadmap.
    flat_roadmap = []
    for aspect in ASPECT_ORDER:
        for suggestion in roadmap.get(aspect, []):
            if suggestion not in flat_roadmap:
                flat_roadmap.append(suggestion)
    if not suggestions:
        suggestions = flat_roadmap[:4]

    return jsonify({
        'sentiment':   {'label': sentiment_pred, 'confidence': confidence},
        'topic':       {'name': aspect_pred},
        'aspects':     aspects,
        'suggestions': suggestions,
        'roadmap':     roadmap,
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
