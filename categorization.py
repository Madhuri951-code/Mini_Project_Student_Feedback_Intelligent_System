
# ==========================================
# STUDENT FEEDBACK INTELLIGENCE SYSTEM
# PHASE 5: FEEDBACK CATEGORIZATION
# ==========================================

import pandas as pd


# Step 1: Define feedback categories and keywords

categories = {

    "Teaching": [
        "teacher", "teachers", "teaching",
        "lecture", "explain", "explains",
        "class", "guidance", "supportive"
    ],

    "Assignments": [
        "assignment", "assignments",
        "homework", "practical",
        "project", "workload"
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


# Step 2: Create categorization function

def categorize_feedback(text):

    text = str(text).lower()

    category_scores = {}

    # Count matching keywords in each category
    for category, keywords in categories.items():

        score = sum(
            1 for word in keywords
            if word in text.split()
        )

        category_scores[category] = score

    # Find the category with the highest score
    best_category = max(
        category_scores,
        key=category_scores.get
    )

    # If no keywords match, assign General Feedback
    if category_scores[best_category] == 0:
        return "General Feedback"

    return best_category


# Step 3: Load sentiment analyzed dataset

df = pd.read_csv("student_feedback_sentiment.csv")


# Step 4: Apply categorization

df["Category"] = df["Cleaned_Feedback"].apply(
    categorize_feedback
)


# Step 5: Display categorization results

print("\n===== STUDENT FEEDBACK CATEGORIZATION =====\n")

print(
    df[["Feedback", "Category"]].to_string(index=False)
)


# Step 6: Calculate category counts

category_counts = df["Category"].value_counts()

print("\n===== CATEGORY SUMMARY =====\n")

print(category_counts)


# Step 7: Calculate category percentages

category_percentages = (
    df["Category"].value_counts(normalize=True) * 100
)

print("\n===== CATEGORY PERCENTAGES =====\n")

print(category_percentages.round(2))


# Step 8: Save categorized dataset

df.to_csv(
    "student_feedback_categorized.csv",
    index=False
)


print("\nCategorization completed successfully!")

print("Total Feedback Records:", len(df))

print("Categorized dataset saved successfully!")
