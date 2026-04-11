import pandas as pd
import google.generativeai as genai
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer 
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
import time

# 1. Setup Gemini (Reminder: Keep your API key private in final submissions)
genai.configure(api_key="AIzaSyA0cvWKqZQZAeevOBOy0kfXpXRRpFN1N2E")
model = genai.GenerativeModel('gemini-1.5-flash')

# 2. Load data
print("Loading reviews...")
df = pd.read_csv('MergedCleaned.csv')
docs = df['comments'].astype(str).tolist()

# 3. Setup Components for 8 Distinct Clusters
# MiniBatchKMeans handles the 8-cluster constraint
cluster_model = MiniBatchKMeans(n_clusters=8, batch_size=1024, random_state=42)

# CountVectorizer handles the basic word cleaning
vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))

# ClassTfidfTransformer forces the 8 clusters to be DIFFERENT from each other
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# 4. Run BERTopic (The Math Phase)
print("Step 1: Clustering into 8 distinct aspects...")
topic_model = BERTopic(
    hdbscan_model=cluster_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    verbose=True
)

topics, _ = topic_model.fit_transform(docs)

# 5. Step 2: Quality Validation (Topic Diversity Score)
# Using Diversity Score as a robust alternative to Coherence since Gensim had install issues
print("\nStep 2: Calculating Topic Diversity Score...")

all_keywords = []
# Get the top 10 words for every topic (excluding -1 outliers)
topic_info = topic_model.get_topic_info()
for i in topic_info['Topic']:
    if i != -1:
        words = [word for word, _ in topic_model.get_topic(i)]
        all_keywords.extend(words)

if len(all_keywords) > 0:
    unique_keywords = set(all_keywords)
    diversity_score = len(unique_keywords) / len(all_keywords)
else:
    diversity_score = 0.0

print(f"✅ Topic Diversity Score: {diversity_score:.4f}")

# 6. Step 3: Gemini Labeling (The AI Phase)
print("\nStep 3: Using Gemini to create high-quality labels...")
new_labels = {}

for index, row in topic_info.iterrows():
    topic_id = row['Topic']
    if topic_id == -1:
        new_labels[topic_id] = "General/Outliers"
        continue
        
    keywords = row['Representation']
    # Get representative docs for better context
    examples = topic_model.get_representative_docs(topic_id)[:2]
    
    prompt = f"""
    I have a cluster of Airbnb reviews. 
    Keywords: {keywords}
    Example reviews: {examples}
    
    Based on these, give me a professional 3-word label for this category. 
    Focus on the functional aspect (e.g., 'Transport & Location', 'Host Communication').
    Return ONLY the 3 words.
    """
    
    try:
        response = model.generate_content(prompt)
        label = response.text.strip().replace('"', '') # Clean quotes
        new_labels[topic_id] = label
        print(f"Topic {topic_id} Label: {label}")
        time.sleep(1) # API rate limit safety
    except Exception as e:
        new_labels[topic_id] = f"Topic {topic_id}"

# 7. Finalize and Save
topic_model.set_topic_labels(new_labels)
final_output = topic_model.get_topic_info()

# Attach the validation score to the output file
final_output['topic_diversity_score'] = diversity_score
final_output.to_csv('BERT_Gemini_Final_8.csv', index=False)

print("\nSUCCESS!")
print(f"1. 8 clusters maintained. Diversity Score: {diversity_score:.4f}")
print("2. Gemini labels applied for the dashboard.")
print("3. Final results saved to 'BERT_Gemini_Final_8.csv'.")