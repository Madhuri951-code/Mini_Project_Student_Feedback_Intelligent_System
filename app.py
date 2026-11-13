
# =====================================================
# STUDENT FEEDBACK INTELLIGENCE SYSTEM
# PHASE 7: INTERACTIVE DASHBOARD
# =====================================================

import streamlit as st
import pandas as pd
import re
import plotly.express as px

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer
)


# -----------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------

st.set_page_config(
    page_title="Student Feedback Intelligence System",
    page_icon="🎓",
    layout="wide"
)


# -----------------------------------------------------
# 2. CUSTOM DASHBOARD TITLE
# -----------------------------------------------------

st.title("🎓 Student Feedback Intelligence System")

st.markdown(
    "### Analyze student opinions using Natural Language Processing"
)

st.write(
    "Upload student feedback to discover sentiments, "
    "categories, frequently mentioned problems and recommendations."
)

st.divider()


# -----------------------------------------------------
# 3. SIDEBAR - DATASET UPLOAD
# -----------------------------------------------------

st.sidebar.header("📂 Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload Student Feedback CSV",
    type=["csv"]
)

st.sidebar.info(
    "Your CSV file must contain a column named Feedback."
)


# -----------------------------------------------------
# 4. LOAD DATASET
# -----------------------------------------------------

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

    except Exception as e:
        st.error(f"Unable to read CSV file: {e}")
        st.stop()

else:

    st.info(
        "No dataset uploaded. Displaying the sample student feedback dataset."
    )

    df = pd.read_csv("student_feedback.csv")


# Validate the dataset

if "Feedback" not in df.columns:

    st.error(
        "Invalid dataset! Please upload a CSV containing a Feedback column."
    )

    st.stop()


# Remove missing and empty feedback

df = df.dropna(subset=["Feedback"]).copy()

df["Feedback"] = df["Feedback"].astype(str)

df = df[
    df["Feedback"].str.strip() != ""
].reset_index(drop=True)


if df.empty:

    st.error("The uploaded dataset contains no valid feedback records.")

    st.stop()


# Create Student_ID if it is missing

if "Student_ID" not in df.columns:

    df.insert(
        0,
        "Student_ID",
        [f"S{i:03d}" for i in range(1, len(df) + 1)]
    )


# -----------------------------------------------------
# 5. TEXT PREPROCESSING
# -----------------------------------------------------

stop_words = set(ENGLISH_STOP_WORDS)

# Preserve negation words

stop_words = stop_words - {
    "not", "no", "nor", "never"
}


def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


df["Cleaned_Feedback"] = df["Feedback"].apply(
    preprocess_text
)


# -----------------------------------------------------
# 6. SENTIMENT ANALYSIS USING VADER
# -----------------------------------------------------

analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text):

    scores = analyzer.polarity_scores(str(text))

    compound = scores["compound"]

    if compound >= 0.05:

        sentiment = "Positive"

    elif compound <= -0.05:

        sentiment = "Negative"

    else:

        sentiment = "Neutral"

    return pd.Series([
        scores["pos"],
        scores["neg"],
        scores["neu"],
        compound,
        sentiment
    ])


df[
    [
        "Positive_Score",
        "Negative_Score",
        "Neutral_Score",
        "Compound_Score",
        "Sentiment"
    ]
] = df["Feedback"].apply(analyze_sentiment)


# -----------------------------------------------------
# 7. FEEDBACK CATEGORIZATION
# -----------------------------------------------------

categories = {

    "Teaching": [
        "teacher", "teachers", "teaching",
        "lecture", "explain", "explains",
        "class", "guidance", "supportive"
    ],

    "Assignments": [
        "assignment", "assignments",
        "homework", "practical",
        "project", "projects", "workload"
    ],

    "Infrastructure": [
        "classroom", "laboratory",
        "equipment", "library",
        "facilities", "infrastructure",
        "computers", "books"
    ],

    "Course Structure": [
        "syllabus", "course",
        "subject", "pace",
        "content", "structured"
    ]
}


def categorize_feedback(text):

    words = str(text).lower().split()

    category_scores = {}

    for category, keywords in categories.items():

        score = sum(
            1 for word in keywords
            if word in words
        )

        category_scores[category] = score

    best_category = max(
        category_scores,
        key=category_scores.get
    )

    if category_scores[best_category] == 0:

        return "General Feedback"

    return best_category


df["Category"] = df["Cleaned_Feedback"].apply(
    categorize_feedback
)


# -----------------------------------------------------
# 8. DASHBOARD OVERVIEW
# -----------------------------------------------------

st.subheader("📊 Feedback Overview")

total_feedback = len(df)

positive_count = (
    df["Sentiment"] == "Positive"
).sum()

negative_count = (
    df["Sentiment"] == "Negative"
).sum()

neutral_count = (
    df["Sentiment"] == "Neutral"
).sum()


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Feedback",
    total_feedback
)

col2.metric(
    "Positive Responses",
    positive_count
)

col3.metric(
    "Negative Responses",
    negative_count
)

col4.metric(
    "Neutral Responses",
    neutral_count
)


st.divider()


# -----------------------------------------------------
# 9. SENTIMENT ANALYSIS VISUALIZATION
# -----------------------------------------------------

st.subheader("😊 Sentiment Analysis")

col1, col2 = st.columns(2)

sentiment_counts = (
    df["Sentiment"]
    .value_counts()
    .reindex(
        ["Positive", "Negative", "Neutral"],
        fill_value=0
    )
)

with col1:

    fig = px.pie(

        names=sentiment_counts.index,

        values=sentiment_counts.values,

        hole=0.55,

        color=sentiment_counts.index,

        color_discrete_map={
            "Positive": "#159D9C",
            "Negative": "#E76B60",
            "Neutral": "#858585"
        },

        title="Distribution of Student Sentiment"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    sentiment_percentage = (
        sentiment_counts / total_feedback * 100
    )

    sentiment_table = pd.DataFrame({

        "Sentiment": sentiment_percentage.index,

        "Percentage": sentiment_percentage.values.round(2)

    })

    st.markdown("#### Sentiment Percentages")

    for _, row in sentiment_table.iterrows():

        st.write(
            f"**{row['Sentiment']}** - "
            f"{row['Percentage']}%"
        )

        st.progress(
            int(row["Percentage"])
        )


st.divider()


# -----------------------------------------------------
# 10. CATEGORY ANALYSIS
# -----------------------------------------------------

st.subheader("📂 Feedback Categories")

category_counts = df["Category"].value_counts()

col1, col2 = st.columns(2)

with col1:

    fig2 = px.bar(

        x=category_counts.index,

        y=category_counts.values,

        labels={
            "x": "Feedback Category",
            "y": "Number of Responses"
        },

        title="Feedback by Category",

        color=category_counts.index

    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


with col2:

    st.markdown("#### Category Distribution")

    category_percentage = (
        category_counts / total_feedback * 100
    )

    for category, percentage in category_percentage.items():

        st.write(
            f"**{category}** - {percentage:.2f}%"
        )

        st.progress(
            int(percentage)
        )


st.divider()


# -----------------------------------------------------
# 11. KEYWORD EXTRACTION
# -----------------------------------------------------

st.subheader("🔑 Frequently Mentioned Keywords")

valid_texts = df["Cleaned_Feedback"]

valid_texts = valid_texts[
    valid_texts.str.strip() != ""
]


keyword_frequency = pd.DataFrame()

tfidf_results = pd.DataFrame()


if len(valid_texts) > 0:

    try:

        # CountVectorizer

        vectorizer = CountVectorizer(

            ngram_range=(1, 2),

            max_features=20

        )

        matrix = vectorizer.fit_transform(valid_texts)

        words = vectorizer.get_feature_names_out()

        counts = matrix.sum(axis=0).A1

        keyword_frequency = pd.DataFrame({

            "Keyword": words,

            "Frequency": counts

        }).sort_values(

            by="Frequency",

            ascending=False

        )


        # TF-IDF

        tfidf_vectorizer = TfidfVectorizer(

            ngram_range=(1, 2),

            max_features=20

        )

        tfidf_matrix = tfidf_vectorizer.fit_transform(
            valid_texts
        )

        tfidf_words = (
            tfidf_vectorizer.get_feature_names_out()
        )

        tfidf_scores = tfidf_matrix.mean(axis=0).A1

        tfidf_results = pd.DataFrame({

            "Keyword": tfidf_words,

            "TF_IDF_Score": tfidf_scores

        }).sort_values(

            by="TF_IDF_Score",

            ascending=False

        )


        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Keyword Frequency")

            fig3 = px.bar(

                keyword_frequency.head(10),

                x="Frequency",

                y="Keyword",

                orientation="h",

                title="Top 10 Frequently Mentioned Keywords"

            )

            fig3.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )


        with col2:

            st.markdown("#### Important Keywords")

            st.dataframe(

                tfidf_results.head(10),

                use_container_width=True,

                hide_index=True

            )


    except ValueError:

        st.warning(
            "Not enough meaningful words for keyword extraction."
        )


st.divider()


# -----------------------------------------------------
# 12. KEY TAKEAWAYS
# -----------------------------------------------------

st.subheader("💡 Key Takeaways")


if positive_count > negative_count:

    st.success(
        "Students have provided more positive than negative feedback."
    )

elif negative_count > positive_count:

    st.warning(
        "Negative feedback is higher. Some areas need improvement."
    )

else:

    st.info(
        "Positive and negative feedback are balanced."
    )


most_discussed = category_counts.idxmax()

st.write(
    f"**Most discussed category:** {most_discussed}"
)


negative_feedback = df[
    df["Sentiment"] == "Negative"
]

if not negative_feedback.empty:

    problem_category = (
        negative_feedback["Category"]
        .value_counts()
    )

    st.write(
        "**Most common negative feedback category:** "
        f"{problem_category.index[0]}"
    )


st.divider()


# -----------------------------------------------------
# 13. RECOMMENDATIONS
# -----------------------------------------------------

st.subheader("🚀 Recommendations")


recommendations = {

    "Teaching":
        "Improve teaching methods, provide clear explanations "
        "and adjust the teaching pace.",

    "Assignments":
        "Provide manageable assignments and introduce more "
        "application-based practical exercises.",

    "Infrastructure":
        "Improve classroom facilities, laboratory equipment "
        "and library resources.",

    "Course Structure":
        "Review the syllabus, course workload and teaching schedule.",

    "General Feedback":
        "Review individual student concerns and provide appropriate support."

}


if negative_feedback.empty:

    st.success(
        "No negative feedback was identified in the current dataset."
    )

else:

    negative_categories = (
        negative_feedback["Category"].value_counts()
    )

    for category, count in negative_categories.items():

        st.markdown(f"**💡 {category}**")

        st.write(recommendations[category])

        st.caption(
            f"{count} negative feedback record(s) in this category."
        )

        st.divider()


# -----------------------------------------------------
# 14. FEEDBACK EXPLORER
# -----------------------------------------------------

st.subheader("🔎 Feedback Explorer")

col1, col2 = st.columns(2)

with col1:

    selected_sentiment = st.selectbox(

        "Filter by Sentiment",

        ["All"] + sorted(df["Sentiment"].unique().tolist())

    )


with col2:

    selected_category = st.selectbox(

        "Filter by Category",

        ["All"] + sorted(df["Category"].unique().tolist())

    )


filtered_df = df.copy()


if selected_sentiment != "All":

    filtered_df = filtered_df[
        filtered_df["Sentiment"] == selected_sentiment
    ]


if selected_category != "All":

    filtered_df = filtered_df[
        filtered_df["Category"] == selected_category
    ]


st.dataframe(

    filtered_df[[
        "Student_ID",
        "Feedback",
        "Sentiment",
        "Category"
    ]],

    use_container_width=True,

    hide_index=True

)

st.write(
    f"Showing {len(filtered_df)} feedback records."
)


st.divider()


# -----------------------------------------------------
# 15. DOWNLOAD REPORTS
# -----------------------------------------------------

st.subheader("📥 Download Reports")

col1, col2, col3 = st.columns(3)


with col1:

    csv_report = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        "Download Complete Report",

        data=csv_report,

        file_name="student_feedback_analysis.csv",

        mime="text/csv"

    )


with col2:

    if not keyword_frequency.empty:

        keyword_csv = keyword_frequency.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(

            "Download Keyword Report",

            data=keyword_csv,

            file_name="keyword_frequency_report.csv",

            mime="text/csv"

        )


with col3:

    if not tfidf_results.empty:

        tfidf_csv = tfidf_results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(

            "Download TF-IDF Report",

            data=tfidf_csv,

            file_name="tfidf_keyword_report.csv",

            mime="text/csv"

        )


# -----------------------------------------------------
# 16. FOOTER
# -----------------------------------------------------

st.divider()

st.caption(
    "Student Feedback Intelligence System | "
    "Developed using Python, NLP, VADER, Scikit-learn and Streamlit"
)

st.caption(
    "Note: Sentiment and category predictions are automated "
    "and may require manual verification."
)
