# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize
import spacy

# ---------------------------
# SETUP
# ---------------------------

st.set_page_config(
    page_title="Argument Strength Analyzer",
    page_icon="🎯",
    layout="wide"
)

# Download punkt if missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None
    st.warning("spaCy model not found. Please install: python -m spacy download en_core_web_sm")


# ---------------------------
# ANALYZER CLASS
# ---------------------------

class ArgumentAnalyzer:

    def analyze_sentence(self, sentence):

        if nlp is None:
            return self.basic_analysis(sentence)
        else:
            return self.advanced_analysis(sentence)

    # -------- BASIC ----------
    def basic_analysis(self, sentence):

        blob = TextBlob(sentence)
        sentiment = blob.sentiment.polarity
        lower = sentence.lower()

        strong_kw = ['evidence','research','study','proven','data']
        weak_kw = ['maybe','perhaps','possibly','might']
        biased_kw = ['always','never','everyone','nobody']
        logical_kw = ['therefore','thus','because','if','then']

        category = "Neutral"
        confidence = 0.5

        if any(k in lower for k in strong_kw):
            category = "Strong"
            confidence = 0.8
        elif any(k in lower for k in weak_kw):
            category = "Weak"
            confidence = 0.7
        elif any(k in lower for k in biased_kw):
            category = "Biased"
            confidence = 0.75
        elif any(k in lower for k in logical_kw):
            category = "Logical"
            confidence = 0.75

        return {
            "sentence": sentence,
            "category": category,
            "confidence": confidence,
            "sentiment": sentiment,
            "word_count": len(sentence.split())
        }

    # -------- ADVANCED NLP ----------
    def advanced_analysis(self, sentence):

        doc = nlp(sentence)

        has_evidence = any(token.text.lower() in ['evidence','study','research','data'] for token in doc)
        has_conclusion = any(token.text.lower() in ['therefore','thus','consequently'] for token in doc)
        has_qualifier = any(token.text.lower() in ['maybe','perhaps','possibly'] for token in doc)
        has_absolute = any(token.text.lower() in ['always','never','everyone'] for token in doc)
        has_number = any(token.like_num for token in doc)

        confidence = 0.6
        category = "Neutral"

        if has_evidence and has_number:
            category = "Strong"
            confidence = 0.95
        elif has_evidence and has_conclusion:
            category = "Strong"
            confidence = 0.9
        elif has_absolute:
            category = "Biased"
            confidence = 0.85
        elif has_qualifier:
            category = "Weak"
            confidence = 0.8
        elif has_conclusion:
            category = "Logical"
            confidence = 0.75

        return {
            "sentence": sentence,
            "category": category,
            "confidence": confidence,
            "word_count": len(sentence.split())
        }

    # -------- ARGUMENT TYPE ----------
    def determine_argument_type(self, text):

        text = text.lower()

        if any(k in text for k in ['history','ancient','century']):
            return "History-based"
        if any(k in text for k in ['research','study','data','experiment']):
            return "Science-based"
        if any(k in text for k in ['i believe','my experience','personally']):
            return "Personal Experience"
        if any(k in text for k in ['therefore','because','if']):
            return "Logical"

        return "General"

    # -------- SUGGESTIONS ----------
    def get_suggestions(self, category):

        suggestions = {
            "Strong": "Great job! You used evidence or strong reasoning.",
            "Weak": "Try adding evidence or stronger wording.",
            "Biased": "Avoid absolute words like 'always' or 'never'.",
            "Logical": "Good reasoning! Add data to make it stronger.",
            "Neutral": "Make your claim clearer and add supporting proof."
        }

        return suggestions.get(category)


# ---------------------------
# UI
# ---------------------------

st.title("🎯 Argument Strength Analyzer")
st.write("Paste your argument below. The system will analyze each sentence using NLP.")

analyzer = ArgumentAnalyzer()

user_input = st.text_area("Enter your argument:", height=200)

if user_input:

    word_count = len(user_input.split())
    sentence_count = len(sent_tokenize(user_input))

    col1, col2 = st.columns(2)
    col1.metric("Words", word_count)
    col2.metric("Sentences", sentence_count)

if st.button("Analyze Argument"):

    if not user_input:
        st.warning("Please enter text.")
        st.stop()

    sentences = sent_tokenize(user_input)
    results = []

    for sentence in sentences:
        results.append(analyzer.analyze_sentence(sentence))

    df = pd.DataFrame(results)

    # Overall Score (0-100)
    overall_score = int(df["confidence"].mean() * 100)

    st.subheader("📊 Overall Result")
    st.metric("Argument Strength Score", f"{overall_score}/100")

    arg_type = analyzer.determine_argument_type(user_input)
    st.info(f"Argument Type: {arg_type}")

    # Chart
    fig = px.bar(
        df,
        x=range(1, len(df)+1),
        y="confidence",
        color="category",
        labels={"x":"Sentence Number","confidence":"Confidence"},
        title="Sentence Confidence Levels"
    )

    st.plotly_chart(fig, width="stretch")

    # Detailed Analysis
    st.subheader("📝 Sentence Analysis")

    for i, row in df.iterrows():

        st.markdown("---")
        st.write(f"**Sentence {i+1}:** {row['sentence']}")
        st.write(f"Category: **{row['category']}**")
        st.write(f"Confidence: **{int(row['confidence']*100)}%**")
        st.write("Suggestion:", analyzer.get_suggestions(row["category"]))

    # Export
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False),
        file_name="argument_analysis.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Built with Streamlit + spaCy NLP")