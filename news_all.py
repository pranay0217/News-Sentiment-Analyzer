import datetime
import requests
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import finnhub as fb
import feedparser


# =======================
# ✅ Load the Sentiment Model
# =======================
json_file = open('sentiment.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('sentiment.h5')
print("\n✅ Loaded model from disk successfully.")

labels = ['negative', 'neutral', 'positive']

# =======================
# ✅ Initialize Finnhub Client (for price info)
# =======================
API_KEY = ''
fb_client = fb.Client(api_key=API_KEY)

# =======================
# ✅ Input Stock Name or Symbol
# =======================
st_name = input("\nEnter the stock name (e.g., AAPL, TSLA, RELIANCE, TATAMOTORS): ").strip()

# =======================
# ✅ Decide Source: 
# - US/Global → Finnhub News
# - India → Google News RSS
# =======================
def is_indian_stock(symbol):
    return symbol.upper().endswith('.NS') or symbol.upper().endswith('.BO') or symbol.upper() in ['RELIANCE', 'TATAMOTORS', 'INFY', 'HDFCBANK']

date_today = datetime.date.today()

if is_indian_stock(st_name):
    # ✅ Use Google News RSS for Indian stocks
    print(f"\n🔍 Fetching news from Google News for {st_name}...")
    rss_query = st_name.replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={rss_query}+stock"
    feed = feedparser.parse(rss_url)

    headlines = [entry.title for entry in feed.entries]

else:
    # ✅ Use Finnhub for global/US stocks
    print(f"\n🔍 Fetching news from Finnhub for {st_name}...")
    try:
        news = fb_client.company_news(st_name, _from='2023-01-01', to=date_today.strftime('%Y-%m-%d'))
        headlines = [article['headline'] for article in news]
    except Exception as e:
        print(f"Error fetching from Finnhub: {e}")
        headlines = []

# =======================
# ✅ Check Headlines Fetched
# =======================
if not headlines:
    print("\n⚠️ No news articles found for this stock.")
    exit()

print(f"\n📰 Found {len(headlines)} headlines.")

# =======================
# ✅ Text Preprocessing
# =======================
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
    text = text.lower().strip()
    return text

preprocessed_headlines = [preprocess_text(h) for h in headlines]

# =======================
# ✅ Tokenization & Padding
# =======================
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(preprocessed_headlines)

sequences = tokenizer.texts_to_sequences(preprocessed_headlines)
padded = pad_sequences(sequences, padding='post', maxlen=30)

# =======================
# ✅ Sentiment Prediction
# =======================
predictions = model.predict(padded)

# =======================
# ✅ Display Results
# =======================
for i, pred in enumerate(predictions):
    headline = headlines[i]
    predicted_index = pred.argmax()
    predicted_label = labels[predicted_index]
    confidence = pred[predicted_index]

    # Colors and emojis
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

    if predicted_label == 'positive':
        color = GREEN
        emoji = '📈 😊'
        expectation = 'Stock price may increase.'
    elif predicted_label == 'neutral':
        color = YELLOW
        emoji = '😐 🔸'
        expectation = 'Stock movement uncertain.'
    else:
        color = RED
        emoji = '📉 😞'
        expectation = 'Stock price may decrease.'

    print("\n" + "=" * 70)
    print(f"{color}Headline: {headline}{RESET}")
    print(f"{color}Sentiment: {predicted_label.upper()} {emoji} (Confidence: {confidence:.2f}){RESET}")
    print(f"{color}{expectation}{RESET}")
    print("=" * 70)


