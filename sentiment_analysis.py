
# ==========================================
# STUDENT FEEDBACK INTELLIGENCE SYSTEM
# PHASE 4: SENTIMENT ANALYSIS USING VADER
# ==========================================

import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Step 1: Initialize VADER
analyzer = SentimentIntensityAnalyzer()


# Step 2: Create sentiment analysis function
def analyze_sentiment(text):

    # Calculate sentiment scores
    scores = analyzer.polarity_scores(str(text))

    # Extract compound score
    compound_score = scores['compound']

    # Classify sentiment
    if compound_score >= 0.05:
        sentiment = "Positive"

    elif compound_score <= -0.05:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return pd.Series([
        scores['pos'],
        scores['neg'],
        scores['neu'],
        compound_score,
        sentiment
    ])


# Step 3: Load cleaned dataset
df = pd.read_csv('student_feedback_cleaned.csv')


# Step 4: Apply sentiment analysis
df[
    ['Positive_Score',
     'Negative_Score',
     'Neutral_Score',
     'Compound_Score',
     'Sentiment']
] = df['Feedback'].apply(analyze_sentiment)


# Step 5: Display sentiment analysis results
print("\n===== STUDENT FEEDBACK SENTIMENT ANALYSIS =====\n")

print(
    df[['Feedback', 'Compound_Score', 'Sentiment']]
    .to_string(index=False)
)


# Step 6: Calculate sentiment counts
sentiment_counts = df['Sentiment'].value_counts()

print("\n===== SENTIMENT SUMMARY =====\n")

print(sentiment_counts)


# Step 7: Calculate sentiment percentages
sentiment_percentages = (
    df['Sentiment'].value_counts(normalize=True) * 100
)

print("\n===== SENTIMENT PERCENTAGES =====\n")

print(sentiment_percentages.round(2))


# Step 8: Save the analyzed dataset
df.to_csv('student_feedback_sentiment.csv', index=False)

print("\nSentiment analysis completed successfully!")

print("Total Feedback Records:", len(df))

print("Analyzed dataset saved successfully!")
