import pandas as pd
from langdetect import detect, LangDetectException, DetectorFactory
import ftfy
import os

# ── Seed langdetect for consistent results ────────────────────────────────────
DetectorFactory.seed = 0

# ── File path — update this to where your merged_reviews.csv is ──────────────
# Option A: put the full path here
# INPUT_PATH = r'C:\Users\Willi\Downloads\merged_reviews.csv'

# Option B: if clean_data.py is in the same folder as merged_reviews.csv,
# just leave it as the filename below and run from that folder
INPUT_PATH = 'merged_reviews.csv'

OUTPUT_PATH = 'mergedcleanedprocess.csv'

# ── Check file exists before loading ─────────────────────────────────────────
if not os.path.exists(INPUT_PATH):
    print(f'❌ File not found: {INPUT_PATH}')
    print(f'   Current working directory: {os.getcwd()}')
    print(f'   Files here: {os.listdir(".")}')
    print('\n   Fix: update INPUT_PATH at the top of this script to the full path of your file.')
    print(r'   Example: INPUT_PATH = r"C:\Users\Willi\Downloads\merged_reviews.csv"')
    exit()

# ── Load merged file ──────────────────────────────────────────────────────────
merged = pd.read_csv(INPUT_PATH)
print('Loaded:', INPUT_PATH)
print('Shape:', merged.shape)

# ── Step 1: ftfy fixes mojibake ───────────────────────────────────────────────
# Repairs garbled text like æˆ¿ → 房 BEFORE language detection
# so langdetect can correctly identify and remove non-English
print('\nStep 1: Fixing encoding with ftfy...')
merged['comments'] = merged['comments'].apply(
    lambda x: ftfy.fix_text(str(x)) if isinstance(x, str) else x
)
print('Done.')

# ── Step 2: langdetect removes all non-English ────────────────────────────────
print('\nStep 2: Detecting languages (may take a few minutes)...')

def detect_lang(text):
    try:
        return detect(str(text))
    except LangDetectException:
        return 'unknown'

merged['lang'] = merged['comments'].apply(detect_lang)
print('Done.')

# ── Language breakdown ────────────────────────────────────────────────────────
print('\nTop 15 detected languages:')
print(merged['lang'].value_counts().head(15))

# ── Step 3: Keep only English ─────────────────────────────────────────────────
cleaned = merged[merged['lang'] == 'en'][
    ['listing_id', 'neighbourhood', 'room_type', 'comments']
].copy()
cleaned = cleaned.reset_index(drop=True)

print(f'\nOriginal rows:  {len(merged):,}')
print(f'Kept (English): {len(cleaned):,}')
print(f'Dropped:        {len(merged) - len(cleaned):,}')

# ── Spot check kept ───────────────────────────────────────────────────────────
print('\n── Sample of KEPT reviews ──')
for _, row in cleaned.sample(5, random_state=42).iterrows():
    print(f'\n  {str(row["comments"])[:200]}')
    print(f'  {"-"*50}')

# ── Spot check dropped ────────────────────────────────────────────────────────
print('\n── Sample of DROPPED reviews ──')
dropped = merged[merged['lang'] != 'en']
for _, row in dropped.sample(5, random_state=42).iterrows():
    print(f'\n  [lang={row["lang"]}] {str(row["comments"])[:200]}')
    print(f'  {"-"*50}')

# ── Export ────────────────────────────────────────────────────────────────────
cleaned.to_csv(OUTPUT_PATH, index=False)

print(f'\n✅ Done!')
print(f'   {OUTPUT_PATH} — {len(cleaned):,} rows, 4 columns')
print(f'   Saved to: {os.path.abspath(OUTPUT_PATH)}')
print('   Upload to Text Miners/Actual Work/data/ on Drive')