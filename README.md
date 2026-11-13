# EDUINSIGHT – Intelligent Student Feedback Analysis and Academic Improvement System

## 📌 Project Overview
EDUINSIGHT is a Python-based Natural Language Processing (NLP) application that analyses student feedback and converts unstructured comments into meaningful insights. It identifies sentiment, categorises feedback, extracts important keywords and presents the results through an interactive Streamlit dashboard.

## 🎯 Objectives
- Analyse student feedback automatically.
- Classify comments into Positive, Negative and Neutral sentiments.
- Categorise feedback into Teaching, Assignments, Infrastructure and Course Structure.
- Extract frequent and important keywords using NLP techniques.
- Generate insights and recommendations for academic improvement.

## 🛠️ Technologies Used
- **Programming Language:** Python
- **NLP:** VADER Sentiment Analysis
- **Text Processing:** NLTK
- **Machine Learning Library:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Keyword Extraction:** CountVectorizer, TF-IDF
- **Dashboard:** Streamlit
- **Visualization:** Plotly

## ⚙️ System Workflow
1. Upload student feedback through a CSV file.
2. Perform text preprocessing and cleaning.
3. Analyse sentiment using VADER.
4. Categorise feedback based on predefined keywords.
5. Extract frequent and distinctive keywords using CountVectorizer and TF-IDF.
6. Display insights through an interactive dashboard.
7. Generate recommendations and downloadable reports.

## 📊 Key Features
- Automated sentiment analysis.
- Feedback categorisation.
- Keyword frequency and TF-IDF analysis.
- Interactive dashboard with visualisations.
- Category-wise insights and recommendations.
- Feedback filtering and CSV report downloads.

## 📁 Project Structure
```text
EDUINSIGHT/
│
├── app.py
├── preprocessing.py
├── sentiment_analysis.py
├── categorization.py
├── keyword_extraction.py
├── student_feedback.csv
├── requirements.txt
└── README.md
```

## 🚀 Installation and Execution

**1. Clone the repository**
```bash
git clone <your-repository-url>
```

**2. Navigate to the project folder**
```bash
cd EDUINSIGHT
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**
```bash
streamlit run app.py
```

## 📈 Sample Results
The system was tested using 40 sample student feedback comments.

- Positive: 55%
- Neutral: 27.5%
- Negative: 17.5%

Assignments and Infrastructure were the most frequently discussed categories.

## 🔮 Future Enhancements
- Transformer-based sentiment analysis.
- Multilingual student feedback support.
- Improved context-aware categorisation.
- Historical feedback and trend analysis.
- Advanced AI-powered recommendations.

## 👩‍💻 Author
**Gundlakunta Madhuri**

Department of Cyber Security and Data Science

Academic Year: 2026–2027

---

*EDUINSIGHT – Turning Student Feedback into Meaningful Academic Insights.*
