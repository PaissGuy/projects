🧠 Sentiment Analysis using RoBERTa and VADER
This project performs sentiment analysis on text data using two popular approaches:

RoBERTa (a transformer-based deep learning model)

VADER (a rule-based model optimized for social media and short text)

📦 Features
Analyze the sentiment of user-provided or batch text input.

Compare and contrast outputs from VADER and RoBERTa.

Visualize sentiment distributions.

Flexible integration with CSV or JSON datasets.

🔍 Models Used
🤖 RoBERTa
Pretrained transformer from Hugging Face

Fine-tuned on Twitter sentiment data.

Provides sentiment probabilities for: Negative, Neutral, and Positive.

Example output:

python
Copy
Edit
{
    'roberta_neg': 0.12,
    'roberta_neu': 0.22,
    'roberta_pos': 0.66,
    'compound': 0.54
}
📝 VADER (Valence Aware Dictionary and sEntiment Reasoner)
Lexicon and rule-based sentiment analysis tool.

Well-suited for social media, news headlines, and conversational text.

Outputs a compound score and polarity breakdown:

python
Copy
Edit
{
    'neg': 0.0,
    'neu': 0.3,
    'pos': 0.7,
    'compound': 0.7269
}

