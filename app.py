import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import spacy
import textstat
from collections import Counter
import re
import nltk
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Argument Strength Analyzer Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# CUSTOM CSS - PREMIUM DARK THEME
# ---------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0F172A;
    }
    
    /* Text colors */
    p, li, span, div {
        color: #F8FAFC !important;
    }
    
    /* Headers gradient */
    h1, h2, h3 {
        background: linear-gradient(90deg, #6366F1, #22D3EE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }
    
    /* Card containers */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div[data-testid="stVerticalBlock"] {
        background-color: #1E293B;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border-left: 4px solid #6366F1;
        margin-bottom: 20px;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #111827;
        border-radius: 12px;
        padding: 15px;
        border-left: 4px solid #6366F1;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: #6366F1 !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: #4F46E5 !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Text area */
    .stTextArea textarea {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #6366F1 !important;
        border-radius: 12px !important;
        font-size: 16px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1E293B;
        padding: 10px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 8px 16px;
        color: #CBD5E1 !important;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #6366F1 !important;
        color: white !important;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    
    .badge-weak { background: #EF4444; color: white; }
    .badge-moderate { background: #F59E0B; color: white; }
    .badge-strong { background: #22C55E; color: white; }
    .badge-research { background: #3B82F6; color: white; }
    
    /* Highlights */
    .highlight-evidence { background-color: #22C55E; color: #0F172A !important; padding: 2px 5px; border-radius: 4px; font-weight: 500; }
    .highlight-logic { background-color: #3B82F6; color: white !important; padding: 2px 5px; border-radius: 4px; font-weight: 500; }
    .highlight-bias { background-color: #FF4D4D; color: white !important; padding: 2px 5px; border-radius: 4px; font-weight: 500; }
    .highlight-weak { background-color: #F97316; color: white !important; padding: 2px 5px; border-radius: 4px; font-weight: 500; }
    .highlight-emotional { background-color: #A855F7; color: white !important; padding: 2px 5px; border-radius: 4px; font-weight: 500; }
    
    /* Info boxes */
    .stAlert {
        background-color: #1E293B !important;
        border-left-color: #6366F1 !important;
        color: #F8FAFC !important;
    }
    
    /* Dividers */
    hr {
        border-color: #6366F1 !important;
        opacity: 0.3;
    }
    
    /* Caption */
    .stCaption {
        color: #CBD5E1 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD MODELS WITH ERROR HANDLING
# ---------------------------
@st.cache_resource
def load_spacy():
    """Load spaCy model with proper error handling"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("""
        ❌ spaCy model 'en_core_web_sm' not found. 
        Please install it using: python -m spacy download en_core_web_sm
        """)
        st.stop()

@st.cache_resource
def load_nltk():
    """Load NLTK data with proper error handling"""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    return True

# Initialize models
nlp = load_spacy()
load_nltk()

# ---------------------------
# OPTIMIZED ARGUMENT ANALYZER CLASS
# ---------------------------
class AdvancedArgumentAnalyzer:
    
    def __init__(self):
        # Feature weights (bounded)
        self.weights = {
            'evidence': 20,
            'statistics': 15,
            'logic': 15,
            'ideal_length': 10,
            'emotional': -10,
            'bias': -20,
            'weak': -15
        }
        
        # Word lists
        self.evidence_words = {'research', 'study', 'data', 'evidence', 'analysis', 'findings', 
                               'experiment', 'survey', 'statistics', 'published', 'journal', 
                               'scientific', 'demonstrated', 'shown'}
        
        self.logic_words = {'therefore', 'thus', 'hence', 'because', 'since', 'consequently',
                           'if', 'then', 'implies', 'due to', 'accordingly', 'furthermore', 
                           'moreover', 'additionally'}
        
        self.bias_words = {'always', 'never', 'everyone', 'nobody', 'all', 'none',
                          'completely', 'absolutely', 'totally', 'undoubtedly',
                          'certainly', 'impossible', 'perfect'}
        
        self.weak_words = {'maybe', 'perhaps', 'possibly', 'might', 'could', 'may',
                          'somewhat', 'probably', 'seems', 'appears', 'suggests'}
        
        self.emotional_words = {'outrageous', 'horrible', 'amazing', 'incredible',
                               'terrible', 'awful', 'fantastic', 'unbelievable',
                               'shocking', 'disgusting', 'wonderful'}
        
        # Improved fallacy detection with controlled lists
        self.ad_hominem_insults = {'stupid', 'idiot', 'foolish', 'ignorant', 'dumb', 
                                   'ridiculous', 'absurd', 'laughable', 'clueless'}
        
        self.generalization_indicators = {'all', 'every', 'none', 'no one', 'everyone', 
                                          'nobody', 'always', 'never'}
        
        self.slippery_slope_phrases = {'lead to', 'result in', 'end with', 'ultimately',
                                       'inevitably', 'certain to'}
        
        self.false_dilemma_phrases = {'either', 'or', 'choose between', 'only two options'}
        
        self.circular_patterns = [r'because.*so', r'so.*because', r'therefore.*because']
        
        # Argument role patterns
        self.role_patterns = {
            'claim': r'\b(argue|believe|think|claim|assert|contend|maintain|propose)\b',
            'evidence': r'\b(according to|research shows|study found|data suggests|evidence indicates|demonstrates that)\b',
            'counterargument': r'\b(however|but|although|yet|nevertheless|on the other hand|conversely|despite)\b',
            'conclusion': r'\b(therefore|thus|hence|consequently|in conclusion|to summarize|overall|ultimately)\b'
        }
    
    def extract_features(self, sentence):
        """Extract linguistic features from sentence (optimized)"""
        doc = nlp(sentence)
        
        # Count occurrences efficiently
        evidence_count = sum(1 for token in doc if token.text.lower() in self.evidence_words)
        logic_count = sum(1 for token in doc if token.text.lower() in self.logic_words)
        bias_count = sum(1 for token in doc if token.text.lower() in self.bias_words)
        weak_count = sum(1 for token in doc if token.text.lower() in self.weak_words)
        emotional_count = sum(1 for token in doc if token.text.lower() in self.emotional_words)
        
        features = {
            'evidence_present': evidence_count > 0,
            'statistics_present': any(token.like_num for token in doc),
            'logic_present': logic_count > 0,
            'emotional_present': emotional_count > 0,
            'bias_present': bias_count > 0,
            'weak_present': weak_count > 0,
            'word_count': len([token for token in doc if not token.is_punct]),
            'has_numbers': any(token.like_num for token in doc),
            'evidence_count': min(evidence_count, 3),  # Cap at 3
            'logic_count': min(logic_count, 3),
            'bias_count': min(bias_count, 3),
            'weak_count': min(weak_count, 3),
            'emotional_count': min(emotional_count, 3)
        }
        
        return features
    
    def calculate_score(self, features):
        """Calculate weighted score for sentence with safe bounds"""
        score = 50.0  # Base score
        
        # Apply positive weights (capped)
        if features['evidence_present']:
            score += self.weights['evidence'] * (features['evidence_count'] / 2)
        if features['statistics_present']:
            score += self.weights['statistics']
        if features['logic_present']:
            score += self.weights['logic'] * (features['logic_count'] / 2)
        
        # Ideal sentence length (10-25 words)
        word_count = features['word_count']
        if 10 <= word_count <= 25:
            score += self.weights['ideal_length']
        elif word_count > 40 or word_count < 5:
            score -= 5
        
        # Apply penalties (capped)
        if features['emotional_present']:
            score += self.weights['emotional'] * (features['emotional_count'] / 2)
        if features['bias_present']:
            score += self.weights['bias'] * (features['bias_count'] / 2)
        if features['weak_present']:
            score += self.weights['weak'] * (features['weak_count'] / 2)
        
        # Safe bounding
        return max(0.0, min(100.0, score))
    
    def get_category(self, score):
        """Get category based on score"""
        if score >= 86:
            return "Research-Level", "🔵"
        elif score >= 71:
            return "Strong", "🟢"
        elif score >= 41:
            return "Moderate", "🟡"
        else:
            return "Weak", "🔴"
    
    def detect_fallacy(self, sentence):
        """Improved fallacy detection with lower false positives"""
        sentence_lower = sentence.lower()
        words = set(sentence_lower.split())
        fallacies = []
        
        # Ad Hominem detection
        if words & self.ad_hominem_insults:
            fallacies.append("Ad Hominem")
        
        # Hasty Generalization
        if any(indicator in sentence_lower for indicator in self.generalization_indicators):
            fallacies.append("Hasty Generalization")
        
        # Slippery Slope
        if any(phrase in sentence_lower for phrase in self.slippery_slope_phrases):
            fallacies.append("Slippery Slope")
        
        # False Dilemma
        if any(phrase in sentence_lower for phrase in self.false_dilemma_phrases):
            if 'either' in sentence_lower and 'or' in sentence_lower:
                fallacies.append("False Dilemma")
        
        # Appeal to Emotion
        if words & self.emotional_words:
            fallacies.append("Appeal to Emotion")
        
        # Circular Reasoning
        for pattern in self.circular_patterns:
            if re.search(pattern, sentence_lower):
                fallacies.append("Circular Reasoning")
                break
        
        return list(set(fallacies)) if fallacies else ["None Detected"]
    
    def determine_argument_role(self, sentence, prev_role=None):
        """Determine the argument role of sentence"""
        sentence_lower = sentence.lower()
        
        for role, pattern in self.role_patterns.items():
            if re.search(pattern, sentence_lower):
                return role.title()
        
        # Default based on position
        if prev_role is None:
            return "Claim"
        elif prev_role.lower() == "claim":
            return "Evidence"
        else:
            return "Conclusion"
    
    @st.cache_data(ttl=3600)
    def get_readability_metrics_cached(_self, text):
        """Cached readability metrics for full text"""
        return {
            'flesch_score': textstat.flesch_reading_ease(text),
            'avg_sentence_length': textstat.avg_sentence_length(text),
            'complex_words': textstat.difficult_words(text),
            'grade_level': textstat.flesch_kincaid_grade(text)
        }
    
    def get_sentence_readability(self, sentence):
        """Lightweight sentence readability approximation"""
        score = textstat.flesch_reading_ease(sentence)
        return score
    
    def get_readability_level(self, score):
        """Determine readability level"""
        if score >= 60:
            return "Easy"
        elif score >= 30:
            return "Moderate"
        else:
            return "Complex"
    
    def generate_rewrite(self, sentence, features):
        """Improved rewrite function with grammar preservation"""
        original = sentence
        
        # Stronger version (remove weak words)
        stronger = sentence
        if features['weak_present']:
            words = sentence.split()
            filtered_words = []
            for word in words:
                clean_word = re.sub(r'[^\w]', '', word.lower())
                if clean_word not in self.weak_words:
                    filtered_words.append(word)
            stronger = ' '.join(filtered_words)
            if not stronger.strip():
                stronger = sentence
        
        # Evidence-based version
        evidence_based = sentence
        if not features['evidence_present'] and len(sentence.split()) > 3:
            evidence_based = f"According to research, {sentence[0].lower()}{sentence[1:]}"
        
        # Neutral version (remove bias and emotional words)
        neutral = sentence
        if features['emotional_present'] or features['bias_present']:
            words = sentence.split()
            filtered_words = []
            for word in words:
                clean_word = re.sub(r'[^\w]', '', word.lower())
                if clean_word not in self.emotional_words and clean_word not in self.bias_words:
                    filtered_words.append(word)
            neutral = ' '.join(filtered_words)
            if not neutral.strip():
                neutral = sentence
        
        return {
            'original': original,
            'stronger': stronger if stronger != original else "Already strong enough",
            'evidence_based': evidence_based,
            'neutral': neutral if neutral != original else "Already neutral"
        }
    
    def analyze_sentence(self, sentence, prev_role=None):
        """Complete analysis of a single sentence"""
        features = self.extract_features(sentence)
        score = self.calculate_score(features)
        category, emoji = self.get_category(score)
        fallacies = self.detect_fallacy(sentence)
        argument_role = self.determine_argument_role(sentence, prev_role)
        readability_score = self.get_sentence_readability(sentence)
        
        return {
            'sentence': sentence,
            'score': round(score, 1),
            'category': category,
            'category_emoji': emoji,
            'confidence': round(score / 100, 2),
            'detected_features': features,
            'fallacies': fallacies,
            'argument_role': argument_role,
            'readability': round(readability_score, 1),
            'readability_level': self.get_readability_level(readability_score)
        }
    
    def generate_debate_arguments(self, text):
        """Generate supporting and counter arguments"""
        preview = text[:100].strip()
        
        return {
            'supporting': f"Supporting this position, {preview.lower()}... This is reinforced by logical reasoning and empirical evidence.",
            'counter': f"However, an alternative perspective suggests that {preview.lower()}... This challenges the initial assumption.",
            'neutral': f"From an academic standpoint, the argument presents key points that require further evidence and consideration."
        }

# ---------------------------
# UI COMPONENTS
# ---------------------------
def display_metric_card(title, value, delta=None, color="#6366F1"):
    """Display a styled metric card"""
    st.markdown(f"""
    <div style="background-color: #111827; padding: 20px; border-radius: 12px; 
                border-left: 4px solid {color}; margin-bottom: 10px;">
        <p style="color: #CBD5E1; margin: 0; font-size: 14px;">{title}</p>
        <p style="color: #F8FAFC; margin: 0; font-size: 28px; font-weight: 700;">{value}</p>
        {f'<p style="color: #22C55E; margin: 0; font-size: 14px;">{delta}</p>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

def display_badge(text, category):
    """Display a colored badge"""
    color_map = {
        'Weak': '#EF4444',
        'Moderate': '#F59E0B',
        'Strong': '#22C55E',
        'Research-Level': '#3B82F6',
        'Claim': '#6366F1',
        'Evidence': '#22C55E',
        'Counterargument': '#F59E0B',
        'Conclusion': '#3B82F6'
    }
    color = color_map.get(category, '#6366F1')
    st.markdown(f"""
    <span style="background-color: {color}; color: white; padding: 4px 12px; 
                  border-radius: 20px; font-size: 12px; font-weight: 600; margin: 2px;">
        {text}
    </span>
    """, unsafe_allow_html=True)

def highlight_text(sentence, analyzer):
    """Improved word highlighting with regex cleaning"""
    words = sentence.split()
    highlighted = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word in analyzer.evidence_words:
            highlighted.append(f'<span class="highlight-evidence">{word}</span>')
        elif clean_word in analyzer.logic_words:
            highlighted.append(f'<span class="highlight-logic">{word}</span>')
        elif clean_word in analyzer.bias_words:
            highlighted.append(f'<span class="highlight-bias">{word}</span>')
        elif clean_word in analyzer.weak_words:
            highlighted.append(f'<span class="highlight-weak">{word}</span>')
        elif clean_word in analyzer.emotional_words:
            highlighted.append(f'<span class="highlight-emotional">{word}</span>')
        else:
            highlighted.append(word)
    
    return ' '.join(highlighted)

# ---------------------------
# MAIN APP
# ---------------------------
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 30px 0;">
        <h1 style="font-size: 48px; margin-bottom: 10px;">🎯 Argument Strength Analyzer Pro</h1>
        <p style="color: #CBD5E1; font-size: 18px;">Research-Grade AI-Powered Debate Intelligence System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = AdvancedArgumentAnalyzer()
    
    # Input section
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        user_input = st.text_area(
            "Enter your argument or debate text:",
            height=200,
            placeholder="Paste your argument here... The system will analyze each sentence for strength, fallacies, and structure.",
            key="input_text"
        )
        
        analyze_btn = st.button("🔍 Analyze Argument", use_container_width=True)
    
    if analyze_btn and user_input:
        with st.spinner("Analyzing argument with advanced AI..."):
            text_to_analyze = user_input
            
            # Tokenize sentences
            sentences = sent_tokenize(text_to_analyze)
            
            # Analyze each sentence
            results = []
            prev_role = None
            
            for sentence in sentences:
                analysis = analyzer.analyze_sentence(sentence, prev_role)
                results.append(analysis)
                prev_role = analysis['argument_role']
            
            df = pd.DataFrame(results)
            
            # Calculate overall metrics
            overall_score = df['score'].mean()
            overall_category, overall_emoji = analyzer.get_category(overall_score)
            
            # Readability metrics for full text (cached)
            readability_metrics = analyzer.get_readability_metrics_cached(text_to_analyze)
            readability_metrics['complex_percentage'] = (
                readability_metrics['complex_words'] / max(textstat.lexicon_count(text_to_analyze, True), 1) * 100
            )
            
            # Tabs for different views
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Overview", "📝 Sentence Analysis", "⚠️ Fallacy Report", 
                "🏗️ Structure Analysis", "📖 Readability", "⚔️ Debate Mode"
            ])
            
            # Tab 1: Overview
            with tab1:
                st.markdown("### Overall Argument Strength")
                
                # Metric cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    color = '#EF4444' if overall_score < 40 else '#F59E0B' if overall_score < 70 else '#22C55E' if overall_score < 86 else '#3B82F6'
                    display_metric_card("Overall Score", f"{overall_score:.1f}/100", f"{overall_category} {overall_emoji}", color)
                
                with col2:
                    avg_length = df['detected_features'].apply(lambda x: x['word_count']).mean()
                    display_metric_card("Total Sentences", len(sentences), f"Avg Length: {avg_length:.0f} words")
                
                with col3:
                    strong_count = len(df[df['category'] == 'Strong'])
                    research_count = len(df[df['category'] == 'Research-Level'])
                    display_metric_card("Strong Arguments", strong_count + research_count, f"{research_count} research-level")
                
                with col4:
                    fallacy_count = sum(1 for f in df['fallacies'] if f != ["None Detected"])
                    display_metric_card("Fallacies Detected", fallacy_count, f"in {len(df)} sentences")
                
                st.markdown("---")
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        x=[f"S{i+1}" for i in range(len(df))],
                        y=df['score'],
                        color=df['category'],
                        color_discrete_map={
                            'Weak': '#EF4444',
                            'Moderate': '#F59E0B',
                            'Strong': '#22C55E',
                            'Research-Level': '#3B82F6'
                        },
                        title="Sentence Strength Scores",
                        labels={'x': 'Sentence', 'y': 'Score'}
                    )
                    fig.update_layout(
                        plot_bgcolor='#1E293B',
                        paper_bgcolor='#1E293B',
                        font_color='#F8FAFC',
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    category_counts = df['category'].value_counts()
                    fig = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title="Argument Distribution",
                        color_discrete_map={
                            'Weak': '#EF4444',
                            'Moderate': '#F59E0B',
                            'Strong': '#22C55E',
                            'Research-Level': '#3B82F6'
                        }
                    )
                    fig.update_layout(
                        plot_bgcolor='#1E293B',
                        paper_bgcolor='#1E293B',
                        font_color='#F8FAFC'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Radar chart with bounded values
                st.markdown("### Multidimensional Analysis")
                
                logic_score = min(100, df['detected_features'].apply(lambda x: x['logic_count']).mean() * 25)
                evidence_score = min(100, df['detected_features'].apply(lambda x: x['evidence_count']).mean() * 25)
                bias_score = min(100, max(0, 100 - (df['detected_features'].apply(lambda x: x['bias_count']).mean() * 30)))
                clarity_score = min(100, df['readability_level'].apply(lambda x: 90 if x == 'Easy' else 60 if x == 'Moderate' else 30).mean())
                emotional_score = min(100, max(0, 100 - (df['detected_features'].apply(lambda x: x['emotional_count']).mean() * 30)))
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=[logic_score, evidence_score, bias_score, clarity_score, emotional_score],
                    theta=['Logic', 'Evidence', 'Low Bias', 'Clarity', 'Low Emotion'],
                    fill='toself',
                    line=dict(color='#6366F1', width=3),
                    fillcolor='rgba(99, 102, 241, 0.3)'
                ))
                
                fig.update_layout(
                    polar=dict(
                        bgcolor='#1E293B',
                        radialaxis=dict(visible=True, range=[0, 100], color='#CBD5E1')
                    ),
                    showlegend=False,
                    paper_bgcolor='#1E293B',
                    font_color='#F8FAFC',
                    title="Argument Quality Radar"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 2: Sentence Analysis
            with tab2:
                st.markdown("### Detailed Sentence Analysis")
                
                for idx, row in df.iterrows():
                    with st.container():
                        st.markdown(f"#### Sentence {idx + 1}")
                        
                        highlighted = highlight_text(row['sentence'], analyzer)
                        st.markdown(f"<div style='background-color: #111827; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>{highlighted}</div>", unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            display_badge(f"{row['category']} {row['category_emoji']}", row['category'])
                        
                        with col2:
                            display_badge(f"Score: {row['score']:.0f}", row['category'])
                        
                        with col3:
                            display_badge(row['argument_role'], row['argument_role'])
                        
                        with col4:
                            display_badge(f"Confidence: {row['confidence']*100:.0f}%", 'Moderate')
                        
                        features = row['detected_features']
                        st.markdown("**Detected Features:**")
                        feature_cols = st.columns(5)
                        
                        with feature_cols[0]:
                            st.markdown(f"📊 Evidence: {'✅' if features['evidence_present'] else '❌'} ({features['evidence_count']})")
                        with feature_cols[1]:
                            st.markdown(f"🔢 Statistics: {'✅' if features['statistics_present'] else '❌'}")
                        with feature_cols[2]:
                            st.markdown(f"🧠 Logic: {'✅' if features['logic_present'] else '❌'} ({features['logic_count']})")
                        with feature_cols[3]:
                            st.markdown(f"⚖️ Bias: {'⚠️' if features['bias_present'] else '✅'} ({features['bias_count']})")
                        with feature_cols[4]:
                            st.markdown(f"😊 Emotional: {'⚠️' if features['emotional_present'] else '✅'} ({features['emotional_count']})")
                        
                        if row['score'] < 70:
                            with st.expander("💡 AI Rewrite Suggestions"):
                                rewrites = analyzer.generate_rewrite(row['sentence'], features)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Original:**")
                                    st.markdown(f"<div style='background-color: #1E293B; padding: 10px; border-radius: 5px;'>{rewrites['original']}</div>", unsafe_allow_html=True)
                                    st.markdown("**Stronger Version:**")
                                    st.markdown(f"<div style='background-color: #1E293B; padding: 10px; border-radius: 5px; border-left: 3px solid #22C55E;'>{rewrites['stronger']}</div>", unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown("**Evidence-Based:**")
                                    st.markdown(f"<div style='background-color: #1E293B; padding: 10px; border-radius: 5px; border-left: 3px solid #3B82F6;'>{rewrites['evidence_based']}</div>", unsafe_allow_html=True)
                                    st.markdown("**Neutral Version:**")
                                    st.markdown(f"<div style='background-color: #1E293B; padding: 10px; border-radius: 5px; border-left: 3px solid #F59E0B;'>{rewrites['neutral']}</div>", unsafe_allow_html=True)
                        
                        st.markdown("---")
            
            # Tab 3: Fallacy Report
            with tab3:
                st.markdown("### Logical Fallacy Detection")
                
                fallacy_df = pd.DataFrame([
                    {
                        'Sentence': f"S{i+1}: {row['sentence'][:100]}...",
                        'Fallacies': ', '.join(row['fallacies'] if row['fallacies'] != ["None Detected"] else ["None"]),
                        'Count': len(row['fallacies']) if row['fallacies'] != ["None Detected"] else 0,
                        'Argument Role': row['argument_role']
                    }
                    for i, row in df.iterrows()
                ])
                
                def color_fallacies(val):
                    if val != 'None':
                        return 'background-color: #EF4444; color: white'
                    return ''
                
                styled_df = fallacy_df.style.map(color_fallacies, subset=['Fallacies'])
                st.dataframe(styled_df, use_container_width=True)
                
                fallacy_counts = Counter()
                for fallacies in df['fallacies']:
                    if fallacies != ["None Detected"]:
                        for f in fallacies:
                            fallacy_counts[f] += 1
                
                if fallacy_counts:
                    st.markdown("### Fallacy Distribution")
                    fig = px.bar(
                        x=list(fallacy_counts.keys()),
                        y=list(fallacy_counts.values()),
                        title="Types of Fallacies Detected",
                        color_discrete_sequence=['#EF4444']
                    )
                    fig.update_layout(
                        plot_bgcolor='#1E293B',
                        paper_bgcolor='#1E293B',
                        font_color='#F8FAFC'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Tab 4: Structure Analysis
            with tab4:
                st.markdown("### Argument Structure Analysis")
                
                role_counts = df['argument_role'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        values=role_counts.values,
                        names=role_counts.index,
                        title="Argument Role Distribution",
                        color_discrete_sequence=['#6366F1', '#22C55E', '#F59E0B', '#3B82F6']
                    )
                    fig.update_layout(
                        plot_bgcolor='#1E293B',
                        paper_bgcolor='#1E293B',
                        font_color='#F8FAFC'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### Argument Flow")
                    flow_data = []
                    for i, row in df.iterrows():
                        flow_data.append({
                            'Position': i+1,
                            'Role': row['argument_role'],
                            'Score': row['score']
                        })
                    
                    flow_df = pd.DataFrame(flow_data)
                    fig = px.line(
                        flow_df,
                        x='Position',
                        y='Score',
                        text='Role',
                        title="Argument Flow and Strength",
                        markers=True
                    )
                    fig.update_traces(line=dict(color='#6366F1', width=3))
                    fig.update_layout(
                        plot_bgcolor='#1E293B',
                        paper_bgcolor='#1E293B',
                        font_color='#F8FAFC'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### Structure Quality")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    claim_score = df[df['argument_role'] == 'Claim']['score'].mean() if 'Claim' in df['argument_role'].values else 0
                    display_metric_card("Claim Strength", f"{claim_score:.1f}/100", "Avg claim score")
                
                with col2:
                    evidence_score = df[df['argument_role'] == 'Evidence']['score'].mean() if 'Evidence' in df['argument_role'].values else 0
                    display_metric_card("Evidence Quality", f"{evidence_score:.1f}/100", "Avg evidence score")
                
                with col3:
                    conclusion_score = df[df['argument_role'] == 'Conclusion']['score'].mean() if 'Conclusion' in df['argument_role'].values else 0
                    display_metric_card("Conclusion Strength", f"{conclusion_score:.1f}/100", "Avg conclusion score")
            
            # Tab 5: Readability
            with tab5:
                st.markdown("### Readability Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    level_color = '#22C55E' if readability_metrics['flesch_score'] >= 60 else '#F59E0B' if readability_metrics['flesch_score'] >= 30 else '#EF4444'
                    display_metric_card(
                        "Flesch Reading Ease", 
                        f"{readability_metrics['flesch_score']:.1f}", 
                        analyzer.get_readability_level(readability_metrics['flesch_score']),
                        level_color
                    )
                
                with col2:
                    display_metric_card(
                        "Grade Level",
                        f"Grade {readability_metrics['grade_level']:.1f}",
                        f"{readability_metrics['avg_sentence_length']:.0f} words/sentence"
                    )
                
                with col3:
                    display_metric_card(
                        "Complex Words",
                        f"{readability_metrics['complex_words']}",
                        f"{readability_metrics['complex_percentage']:.1f}% of text"
                    )
                
                st.markdown("### Sentence Readability Breakdown")
                
                readability_df = pd.DataFrame([
                    {
                        'Sentence': f"S{i+1}",
                        'Score': row['readability'],
                        'Level': row['readability_level'],
                        'Words': row['detected_features']['word_count']
                    }
                    for i, row in df.iterrows()
                ])
                
                fig = px.bar(
                    readability_df,
                    x='Sentence',
                    y='Score',
                    color='Level',
                    color_discrete_map={
                        'Easy': '#22C55E',
                        'Moderate': '#F59E0B',
                        'Complex': '#EF4444'
                    },
                    title="Readability Scores by Sentence"
                )
                fig.update_layout(
                    plot_bgcolor='#1E293B',
                    paper_bgcolor='#1E293B',
                    font_color='#F8FAFC'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if readability_metrics['flesch_score'] < 30:
                    st.warning("⚠️ Text is complex. Consider simplifying sentence structure and using shorter words.")
                elif readability_metrics['flesch_score'] < 60:
                    st.info("📝 Text has moderate readability. Could be improved for wider audience.")
                else:
                    st.success("✅ Text is easily readable and accessible!")
            
            # Tab 6: Debate Mode
            with tab6:
                st.markdown("### Debate Mode")
                st.markdown("Generate supporting and counter arguments based on your input.")
                
                debate_args = analyzer.generate_debate_arguments(text_to_analyze)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### ⚔️ Supporting Argument")
                    st.markdown(f"""
                    <div style="background-color: #111827; padding: 20px; border-radius: 12px; 
                                border-left: 4px solid #22C55E; margin-bottom: 20px;">
                        <p style="color: #F8FAFC;">{debate_args['supporting']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### 🛡️ Counter Argument")
                    st.markdown(f"""
                    <div style="background-color: #111827; padding: 20px; border-radius: 12px; 
                                border-left: 4px solid #EF4444; margin-bottom: 20px;">
                        <p style="color: #F8FAFC;">{debate_args['counter']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### 📚 Neutral Academic Summary")
                    st.markdown(f"""
                    <div style="background-color: #111827; padding: 20px; border-radius: 12px; 
                                border-left: 4px solid #3B82F6; margin-bottom: 20px;">
                        <p style="color: #F8FAFC;">{debate_args['neutral']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### 💡 Debate Tips")
                    st.markdown("""
                    - **Support claims** with evidence and data
                    - **Address counterarguments** to strengthen position
                    - **Use logical connectors** for clarity
                    - **Avoid emotional language** in academic debate
                    - **Consider multiple perspectives** for balanced argument
                    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #CBD5E1; padding: 20px;">
        <p>🎯 Advanced Argument Strength Analyzer Pro | Research-Grade AI-Powered Debate Intelligence</p>
        <p style="font-size: 12px;">Built with Streamlit, spaCy, NLTK, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()