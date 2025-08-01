import datetime
import requests
import re
import feedparser
import finnhub as fb
from keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# =======================
# ✅ Load Sentiment Model
# =======================
json_file = open('sentiment.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('sentiment.h5')
print("\n✅ Loaded sentiment model successfully.")

labels = ['negative', 'neutral', 'positive']

# =======================
# ✅ Initialize Finnhub API
# =======================
API_KEY = 'd1g258pr01qk4ao0inggd1g258pr01qk4ao0inh0'
fb_client = fb.Client(api_key=API_KEY)

# =======================
# ✅ Helper Functions
# =======================

# ✔️ Check if the stock is Indian
def is_indian_stock(symbol):
    return symbol.upper().endswith('.NS') or symbol.upper().endswith('.BO') or symbol.upper() in ['RELIANCE', 'TATAMOTORS', 'INFY', 'HDFCBANK']

# ✔️ Preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
    text = text.lower().strip()
    return text

# ✔️ Color console output
def get_color_and_emoji(label):
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

    if label == 'positive':
        return GREEN, '📈 😊', 'Stock likely to rise.', RESET
    elif label == 'neutral':
        return YELLOW, '😐 🔸', 'Stock movement uncertain.', RESET
    else:
        return RED, '📉 😞', 'Stock likely to fall.', RESET

# ✔️ Display overall sentiment
def overall_sentiment(predictions):
    counts = {'positive': 0, 'neutral': 0, 'negative': 0}

    for pred in predictions:
        idx = pred.argmax()
        counts[labels[idx]] += 1

    result = max(counts, key=counts.get)
    return result, counts


# =======================
# ✅ Main Script
# =======================

# ✔️ Input stock name
st_name = input("\nEnter stock symbol or name (e.g., AAPL, TSLA, RELIANCE, RELIANCE.NS): ").strip()

# ✔️ Get today's date
date_today = datetime.date.today()

# ✔️ Fetch news
if is_indian_stock(st_name):
    print(f"\n🔍 Fetching news for {st_name} from Google News RSS...")
    rss_query = st_name.replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={rss_query}+stock"
    feed = feedparser.parse(rss_url)

    headlines = [entry.title for entry in feed.entries]

else:
    print(f"\n🔍 Fetching news for {st_name} from Finnhub...")
    try:
        news = fb_client.company_news(st_name, _from='2023-01-01', to=date_today.strftime('%Y-%m-%d'))
        headlines = [article['headline'] for article in news]
    except Exception as e:
        print(f"Error fetching from Finnhub: {e}")
        headlines = []

# ✔️ Check headlines
if not headlines:
    print("\n⚠️ No news articles found for this stock.")
    exit()

print(f"\n📰 Found {len(headlines)} headlines.")

# ✔️ Preprocess headlines
preprocessed_headlines = [preprocess_text(h) for h in headlines]

# ✔️ Tokenization and padding
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(preprocessed_headlines)

sequences = tokenizer.texts_to_sequences(preprocessed_headlines)
padded = pad_sequences(sequences, padding='post', maxlen=30)

# ✔️ Sentiment prediction
predictions = model.predict(padded)

# ✔️ Display headline-wise sentiment
for i, pred in enumerate(predictions):
    headline = headlines[i]
    predicted_index = pred.argmax()
    predicted_label = labels[predicted_index]
    confidence = pred[predicted_index]

    color, emoji, expectation, RESET = get_color_and_emoji(predicted_label)

    print("\n" + "=" * 70)
    print(f"{color}Headline: {headline}{RESET}")
    print(f"{color}Sentiment: {predicted_label.upper()} {emoji} (Confidence: {confidence:.2f}){RESET}")
    print(f"{color}{expectation}{RESET}")
    print("=" * 70)

# ✔️ Compute overall sentiment
overall, counts = overall_sentiment(predictions)

print("\n\n" + "#" * 80)
print(f"\n📊 Overall Market Sentiment for {st_name.upper()}:")
print(f"Positive: {counts['positive']} | Neutral: {counts['neutral']} | Negative: {counts['negative']}")
print(f"🎯 Final Sentiment: {overall.upper()}")
print("#" * 80)
