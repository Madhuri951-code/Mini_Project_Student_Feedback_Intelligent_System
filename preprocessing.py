
# Import required libraries

import re
import pandas as pd

from nltk.corpus import stopwords


# Load English stopwords
stop_words = set(stopwords.words('english'))

# Preserve negation words because they affect sentiment
negation_words = {'not', 'no', 'nor', 'never'}

stop_words = stop_words - negation_words


# Text preprocessing function
def preprocess_text(text):

    # 1. Convert text into lowercase
    text = text.lower()

    # 2. Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Tokenization
    words = text.split()

    # 4. Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    # 5. Join words into a clean sentence
    cleaned_text = ' '.join(words)

    return cleaned_text


# Load student feedback dataset
df = pd.read_csv('student_feedback.csv')


# Apply preprocessing to every feedback record
df['Cleaned_Feedback'] = df['Feedback'].apply(
    preprocess_text
)


# Display original and cleaned feedback
print("\nSTUDENT FEEDBACK PREPROCESSING RESULTS\n")

print(df[['Feedback', 'Cleaned_Feedback']].head(10))


# Save the cleaned dataset
df.to_csv('student_feedback_cleaned.csv', index=False)

print("\nPreprocessing completed successfully!")
print("Total feedback records:", len(df))
print("Cleaned dataset saved successfully!")
