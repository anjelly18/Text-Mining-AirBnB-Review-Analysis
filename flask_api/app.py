import os
import re
import pickle
import nltk
import ftfy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify
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

# ── Load models ───────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'Actual Work', 'models')

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

# ── API endpoint ──────────────────────────────────────────────────────────────
@app.route('/analyse', methods=['POST'])
def analyse():
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

    # Task 3: aspect NSS scores (rule-based, real VADER)
    aspects     = compute_nss(raw_text)

    # Task 3: improvement signals
    suggestions = extract_suggestions(raw_text)

    return jsonify({
        'sentiment':   {'label': sentiment_pred, 'confidence': confidence},
        'topic':       {'name': aspect_pred},
        'aspects':     aspects,
        'suggestions': suggestions,
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
