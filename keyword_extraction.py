
# ==========================================
# STUDENT FEEDBACK INTELLIGENCE SYSTEM
# PHASE 6: KEYWORD EXTRACTION
# ==========================================

import pandas as pd

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)


# Step 1: Load categorized dataset

df = pd.read_csv("student_feedback_categorized.csv")


# Step 2: Prepare cleaned feedback text

texts = df["Cleaned_Feedback"].fillna("").astype(str)

# Remove empty feedback records
texts = texts[texts.str.strip() != ""]


# Step 3: Extract keywords using CountVectorizer

count_vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    max_features=30
)

count_matrix = count_vectorizer.fit_transform(texts)


# Step 4: Calculate word and phrase frequencies

words = count_vectorizer.get_feature_names_out()

frequencies = count_matrix.sum(axis=0).A1


keyword_frequency = pd.DataFrame({
    "Keyword": words,
    "Frequency": frequencies
})

keyword_frequency = keyword_frequency.sort_values(
    by="Frequency",
    ascending=False
)


# Step 5: Display frequently mentioned keywords

print("\n===== FREQUENTLY MENTIONED KEYWORDS =====\n")

print(keyword_frequency.to_string(index=False))


# Step 6: Extract important keywords using TF-IDF

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=30
)

tfidf_matrix = tfidf_vectorizer.fit_transform(texts)


# Step 7: Calculate average TF-IDF scores

terms = tfidf_vectorizer.get_feature_names_out()

average_scores = tfidf_matrix.mean(axis=0).A1

tfidf_results = pd.DataFrame({
    "Keyword": terms,
    "TF_IDF_Score": average_scores
})

tfidf_results = tfidf_results.sort_values(
    by="TF_IDF_Score",
    ascending=False
)


# Step 8: Display important keywords

print("\n===== IMPORTANT KEYWORDS USING TF-IDF =====\n")

print(tfidf_results.to_string(index=False))


# Step 9: Save keyword frequency report

keyword_frequency.to_csv(
    "keyword_frequency_report.csv",
    index=False
)


# Step 10: Save TF-IDF report

tfidf_results.to_csv(
    "tfidf_keyword_report.csv",
    index=False
)


# Step 11: Generate category-wise keyword reports

print("\n===== CATEGORY-WISE KEYWORD ANALYSIS =====\n")

category_reports = []

for category in df["Category"].dropna().unique():

    category_texts = df.loc[
        df["Category"] == category,
        "Cleaned_Feedback"
    ].dropna()

    category_texts = category_texts[
        category_texts.str.strip() != ""
    ]

    if len(category_texts) == 0:
        continue

    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        max_features=10
    )

    try:
        matrix = vectorizer.fit_transform(category_texts)

        category_words = vectorizer.get_feature_names_out()

        category_counts = matrix.sum(axis=0).A1

        for word, count in zip(category_words, category_counts):
            category_reports.append({
                "Category": category,
                "Keyword": word,
                "Frequency": count
            })

        print(f"\n{category}")
        print(
            pd.DataFrame({
                "Keyword": category_words,
                "Frequency": category_counts
            }).sort_values(
                by="Frequency",
                ascending=False
            ).to_string(index=False)
        )

    except ValueError:
        print(f"Not enough vocabulary for {category}")


# Step 12: Save category-wise keyword report

category_df = pd.DataFrame(category_reports)

category_df.to_csv(
    "category_keyword_report.csv",
    index=False
)


print("\nKeyword extraction completed successfully!")

print("Total feedback records analyzed:", len(texts))

print("All keyword reports saved successfully!")
