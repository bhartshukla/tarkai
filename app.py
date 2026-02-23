import streamlit as st
import pandas as pd
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
    page_title="tarkAI Analyzer Pro | Legal Academic Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# CUSTOM CSS - PROFESSIONAL ACADEMIC THEME
# ---------------------------
def get_theme_css(academic_mode=False):
    if academic_mode:
        return """
        <style>
            /* Light academic theme */
            .stApp { background-color: #FFFFFF; }
            p, li, span, div { color: #111111 !important; }
            h1, h2, h3 { color: #2C3E50 !important; background: none; -webkit-text-fill-color: #2C3E50; }
            div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div[data-testid="stVerticalBlock"] {
                background-color: #F5F5F5; border-radius: 8px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); border-left: 4px solid #2C3E50;
            }
            div[data-testid="metric-container"] { background-color: #F5F5F5; border-radius: 8px; padding: 15px; border-left: 4px solid #2C3E50; }
            .stButton > button { background: #2C3E50 !important; color: white !important; border-radius: 6px !important; border: none !important; }
            .stTextArea textarea { background-color: #F5F5F5 !important; color: #111111 !important; border: 1px solid #2C3E50 !important; border-radius: 8px !important; }
            .stTabs [data-baseweb="tab-list"] { background-color: #F5F5F5; }
            .stTabs [aria-selected="true"] { background-color: #2C3E50 !important; color: white !important; }
            .badge-weak { background: #EF4444; color: white; }
            .badge-moderate { background: #F59E0B; color: white; }
            .badge-strong { background: #22C55E; color: white; }
            .badge-research { background: #2C3E50; color: white; }
            .highlight-evidence { background-color: #22C55E; color: white !important; padding: 2px 5px; border-radius: 4px; }
            .highlight-logic { background-color: #2C3E50; color: white !important; padding: 2px 5px; border-radius: 4px; }
            .highlight-bias { background-color: #EF4444; color: white !important; padding: 2px 5px; border-radius: 4px; }
            .stAlert { background-color: #F5F5F5 !important; border-left-color: #2C3E50 !important; }
        </style>
        """
    else:
        return """
        <style>
            .stApp { background-color: #0F172A; }
            p, li, span, div { color: #F8FAFC !important; }
            h1, h2, h3 { background: linear-gradient(90deg, #6366F1, #22D3EE); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700 !important; }
            div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div[data-testid="stVerticalBlock"] {
                background-color: #1E293B; border-radius: 15px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); border-left: 4px solid #6366F1; margin-bottom: 20px;
            }
            div[data-testid="metric-container"] { background-color: #111827; border-radius: 12px; padding: 15px; border-left: 4px solid #6366F1; }
            .stButton > button { background: #6366F1 !important; color: white !important; border-radius: 10px !important; border: none !important; }
            .stTextArea textarea { background-color: #1E293B !important; color: #F8FAFC !important; border: 1px solid #6366F1 !important; border-radius: 12px !important; }
            .stTabs [data-baseweb="tab-list"] { background-color: #1E293B; }
            .stTabs [aria-selected="true"] { background-color: #6366F1 !important; color: white !important; }
            .badge-weak { background: #EF4444; color: white; }
            .badge-moderate { background: #F59E0B; color: white; }
            .badge-strong { background: #22C55E; color: white; }
            .badge-research { background: #3B82F6; color: white; }
            .highlight-evidence { background-color: #22C55E; color: #0F172A !important; padding: 2px 5px; border-radius: 4px; }
            .highlight-logic { background-color: #3B82F6; color: white !important; padding: 2px 5px; border-radius: 4px; }
            .highlight-bias { background-color: #EF4444; color: white !important; padding: 2px 5px; border-radius: 4px; }
            .stAlert { background-color: #1E293B !important; border-left-color: #6366F1 !important; }
        </style>
        """

# ---------------------------
# LOAD MODELS WITH ERROR HANDLING
# ---------------------------
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("Please install: python -m spacy download en_core_web_sm")
        st.stop()

@st.cache_resource
def load_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    return True

nlp = load_spacy()
load_nltk()

# ---------------------------
# LEGAL ANALYZER MODULE
# ---------------------------
class LegalAnalyzer:
    def __init__(self):
        # IRAC detection patterns
        self.irac_patterns = {
            'issue': [
                r'\b(whether|issue is|the question|does|can|should)\b',
                r'\b(who|what|when|where|why|how)\b.*\?',
                r'\bthe (primary|main|central) (issue|question)\b'
            ],
            'rule': [
                r'\b(according to|section|article|statute|under the law)\b',
                r'\bas held in|as stated in|the court in|precedent|doctrine\b',
                r'\brule|principle|doctrine|established|jurisprudence\b'
            ],
            'application': [
                r'\bin this case|applying|here|the facts show|in the present case\b',
                r'\btherefore,? .+ (because|since|as)\b',
                r'\bapplying the (rule|principle|test) to\b'
            ],
            'conclusion': [
                r'\btherefore|thus|hence|in conclusion|consequently\b',
                r'\bfor these reasons|accordingly|in light of the above\b',
                r'\bwe hold|it is held|the court (finds|concludes)\b'
            ]
        }
        
        # Formality indicators
        self.formal_indicators = {
            'passive_voice': [r'\b(is|are|was|were|be|been|being) \w+ed\b'],
            'objective_tone': [r'\b(it is|there is|one might|it can be)\b'],
            'formal_vocab': [
                r'\b(herein|thereof|thereto|aforesaid|hereinafter)\b',
                r'\b(pursuant|thereunder|thereby|therewith)\b'
            ],
            'no_personal': [r'\b(I|me|my|mine|we|us|our|yours?)\b']
        }
        
        # Informal/emotional flags
        self.informal_phrases = [
            r'\b(literally|basically|actually|honestly|totally)\b',
            r'\b(kind of|sort of|a bit|a lot)\b',
            r'\b(amazing|awful|terrible|great|bad|good|nice)\b'
        ]
        
        self.absolute_claims = [
            r'\b(always|never|everyone|nobody|all|none|every|no one)\b',
            r'\b(completely|absolutely|totally|undoubtedly|certainly)\b'
        ]
        
        # Transition words for coherence
        self.transition_words = {
            'addition': ['furthermore', 'moreover', 'additionally', 'also'],
            'contrast': ['however', 'but', 'although', 'nevertheless', 'conversely'],
            'causal': ['therefore', 'thus', 'hence', 'consequently', 'accordingly'],
            'sequence': ['first', 'second', 'third', 'finally', 'next']
        }
        
    def detect_irac_structure(self, sentences):
        """Classify sentences by IRAC role"""
        results = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            role = 'other'
            confidence = 0.0
            
            for irac_role, patterns in self.irac_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, sentence_lower):
                        role = irac_role
                        confidence = 0.8
                        break
                if role != 'other':
                    break
            
            results.append({
                'sentence': sentence,
                'role': role,
                'confidence': confidence
            })
        
        # Check completeness
        roles_found = {r['role'] for r in results if r['role'] != 'other'}
        missing = [r for r in ['issue', 'rule', 'application', 'conclusion'] if r not in roles_found]
        
        return results, missing
    
    def calculate_formality_score(self, text):
        """Calculate formality score (0-100)"""
        score = 100
        text_lower = text.lower()
        
        # Penalize personal pronouns
        personal_count = len(re.findall(r'\b(I|me|my|mine|we|us|our|yours?)\b', text_lower))
        score -= personal_count * 5
        
        # Reward formal vocabulary
        formal_count = len(re.findall(r'\b(herein|thereof|pursuant|aforesaid|hereinafter)\b', text_lower))
        score += formal_count * 3
        
        # Penalize informal phrases
        for pattern in self.informal_phrases:
            if re.search(pattern, text_lower):
                score -= 10
        
        # Check passive voice (formal indicator)
        passive_count = len(re.findall(r'\b(is|are|was|were) \w+ed\b', text_lower))
        score += passive_count * 2
        
        return max(0, min(100, score))
    
    def check_legal_quality(self, sentences):
        """Flag inappropriate legal writing"""
        flags = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check informality
            for pattern in self.informal_phrases:
                if re.search(pattern, sentence_lower):
                    flags.append({
                        'sentence': sentence,
                        'issue': 'Informal language',
                        'suggestion': 'Use formal academic vocabulary'
                    })
                    break
            
            # Check absolute claims
            for pattern in self.absolute_claims:
                if re.search(pattern, sentence_lower):
                    flags.append({
                        'sentence': sentence,
                        'issue': 'Absolute claim without qualification',
                        'suggestion': 'Qualify with "generally", "typically", or cite authority'
                    })
                    break
            
            # Check emotional tone
            emotional = ['outrageous', 'horrible', 'shocking', 'disgusting']
            if any(word in sentence_lower for word in emotional):
                flags.append({
                    'sentence': sentence,
                    'issue': 'Emotional tone',
                    'suggestion': 'Maintain objective, dispassionate tone'
                })
        
        return flags
    
    def suggest_citation(self, sentences):
        """Suggest citations for unsupported claims"""
        suggestions = []
        claim_patterns = [
            r'\b(is|are|was|were) the (most|best|worst)\b',
            r'\b(leads to|causes|results in)\b',
            r'\b(proves|demonstrates|shows) that\b'
        ]
        
        for sentence in sentences:
            if len(sentence.split()) > 10:  # Only check substantial sentences
                for pattern in claim_patterns:
                    if re.search(pattern, sentence.lower()):
                        suggestions.append({
                            'sentence': sentence,
                            'suggestion': 'Consider adding case law or academic citation'
                        })
                        break
        
        return suggestions
    
    def calculate_coherence_score(self, sentences):
        """Calculate argument coherence (0-100)"""
        if len(sentences) < 2:
            return 100
        
        score = 70  # Base score
        text_lower = ' '.join(sentences).lower()
        
        # Check transition words
        transition_count = 0
        for category, words in self.transition_words.items():
            for word in words:
                if word in text_lower:
                    transition_count += 1
        
        # Ideal transitions per 5 sentences
        expected = len(sentences) / 5
        transition_score = min(30, (transition_count / max(expected, 1)) * 30)
        score += transition_score
        
        # Check topic consistency (simplified - uses keyword overlap)
        keywords = []
        for sent in sentences[:3]:  # First few sentences establish topic
            words = re.findall(r'\b[a-z]{4,}\b', sent.lower())
            keywords.extend(words)
        
        keyword_set = set(keywords[:5])
        if keyword_set:
            consistent = 0
            for sent in sentences[1:]:
                if any(k in sent.lower() for k in keyword_set):
                    consistent += 1
            consistency_score = (consistent / max(len(sentences)-1, 1)) * 20
            score += consistency_score
        
        return max(0, min(100, score))

# ---------------------------
# TOXICITY ANALYZER MODULE
# ---------------------------
class ToxicityAnalyzer:
    def __init__(self):
        self.toxic_patterns = {
            'personal_attack': [
                r'\b(you are (stupid|idiot|fool|dumb|ignorant))\b',
                r'\b(you (always|never) .+ (wrong|incorrect))\b',
                r'\b(your argument|your point) is (ridiculous|absurd|laughable)\b'
            ],
            'insults': [
                r'\b(stupid|idiot|moron|foolish|dumb|ignorant|pathetic)\b',
                r'\b(wrong|incorrect|false) (as always|as usual)\b'
            ],
            'hate_speech_indicators': [
                r'\b(these people|those people|they (always|never))\b',
                r'\b(against (common sense|decency|reason))\b'
            ],
            'aggressive_language': [
                r'\b(shut up|be quiet|nonsense|rubbish)\b',
                r'\b(how dare you|you must|you have to)\b'
            ]
        }
        
        self.shouting_pattern = r'[A-Z]{4,}'  # Words in all caps
        
    def analyze_toxicity(self, text):
        """Full toxicity analysis"""
        text_lower = text.lower()
        words = text.split()
        
        # Check for shouting
        shouting_count = len(re.findall(self.shouting_pattern, text))
        shouting_score = min(30, shouting_count * 10)
        
        # Check toxic patterns
        toxic_matches = []
        toxicity_score = 0
        
        for category, patterns in self.toxic_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    toxic_matches.extend([(category, m) for m in matches])
                    toxicity_score += len(matches) * 15
        
        # Add shouting to toxicity
        toxicity_score += shouting_score
        
        # Bound score
        toxicity_score = max(0, min(100, toxicity_score))
        
        # Categorize
        if toxicity_score <= 20:
            category = "Clean"
            color = "#22C55E"
        elif toxicity_score <= 50:
            category = "Mildly Aggressive"
            color = "#F59E0B"
        elif toxicity_score <= 75:
            category = "Toxic"
            color = "#EF4444"
        else:
            category = "Highly Abusive"
            color = "#7F1D1D"
        
        # Find abusive words to highlight
        abusive_words = []
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            for _, insults in self.toxic_patterns.items():
                if any(re.search(rf'\b{re.escape(clean_word)}\b', insult) for insult in insults):
                    abusive_words.append(word)
                    break
        
        return {
            'score': toxicity_score,
            'category': category,
            'color': color,
            'shouting_count': shouting_count,
            'toxic_matches': toxic_matches,
            'abusive_words': list(set(abusive_words)),
            'suggestion': "Rewrite in neutral academic tone" if toxicity_score > 60 else None
        }
    
    def highlight_toxic_text(self, text, abusive_words):
        """Return HTML with toxic words highlighted"""
        if not abusive_words:
            return text
        
        highlighted = text
        for word in abusive_words:
            pattern = rf'\b{re.escape(word)}\b'
            replacement = f'<span style="background-color: #EF4444; color: white; padding: 2px 5px; border-radius: 4px;">{word}</span>'
            highlighted = re.sub(pattern, replacement, highlighted, flags=re.IGNORECASE)
        
        return highlighted

# ---------------------------
# CORE ANALYZER MODULE (OPTIMIZED)
# ---------------------------
class CoreAnalyzer:
    def __init__(self):
        self.weights = {
            'evidence': 20, 'statistics': 15, 'logic': 15,
            'ideal_length': 10, 'emotional': -10, 'bias': -20, 'weak': -15
        }
        
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
        
        self.ad_hominem_insults = {'stupid', 'idiot', 'foolish', 'ignorant', 'dumb', 
                                   'ridiculous', 'absurd', 'laughable', 'clueless'}
        self.generalization_indicators = {'all', 'every', 'none', 'no one', 'everyone', 
                                          'nobody', 'always', 'never'}
        self.slippery_slope_phrases = {'lead to', 'result in', 'end with', 'ultimately',
                                       'inevitably', 'certain to'}
        self.false_dilemma_phrases = {'either', 'or', 'choose between', 'only two options'}
        self.circular_patterns = [r'because.*so', r'so.*because', r'therefore.*because']
        
        self.role_patterns = {
            'claim': r'\b(argue|believe|think|claim|assert|contend|maintain|propose)\b',
            'evidence': r'\b(according to|research shows|study found|data suggests|evidence indicates|demonstrates that)\b',
            'counterargument': r'\b(however|but|although|yet|nevertheless|on the other hand|conversely|despite)\b',
            'conclusion': r'\b(therefore|thus|hence|consequently|in conclusion|to summarize|overall|ultimately)\b'
        }
    
    def extract_features(self, sentence):
        doc = nlp(sentence)
        evidence_count = sum(1 for token in doc if token.text.lower() in self.evidence_words)
        logic_count = sum(1 for token in doc if token.text.lower() in self.logic_words)
        bias_count = sum(1 for token in doc if token.text.lower() in self.bias_words)
        weak_count = sum(1 for token in doc if token.text.lower() in self.weak_words)
        emotional_count = sum(1 for token in doc if token.text.lower() in self.emotional_words)
        
        return {
            'evidence_present': evidence_count > 0,
            'statistics_present': any(token.like_num for token in doc),
            'logic_present': logic_count > 0,
            'emotional_present': emotional_count > 0,
            'bias_present': bias_count > 0,
            'weak_present': weak_count > 0,
            'word_count': len([token for token in doc if not token.is_punct]),
            'has_numbers': any(token.like_num for token in doc),
            'evidence_count': min(evidence_count, 3),
            'logic_count': min(logic_count, 3),
            'bias_count': min(bias_count, 3),
            'weak_count': min(weak_count, 3),
            'emotional_count': min(emotional_count, 3)
        }
    
    def calculate_score(self, features):
        score = 50.0
        if features['evidence_present']:
            score += self.weights['evidence'] * (features['evidence_count'] / 2)
        if features['statistics_present']:
            score += self.weights['statistics']
        if features['logic_present']:
            score += self.weights['logic'] * (features['logic_count'] / 2)
        
        word_count = features['word_count']
        if 10 <= word_count <= 25:
            score += self.weights['ideal_length']
        elif word_count > 40 or word_count < 5:
            score -= 5
        
        if features['emotional_present']:
            score += self.weights['emotional'] * (features['emotional_count'] / 2)
        if features['bias_present']:
            score += self.weights['bias'] * (features['bias_count'] / 2)
        if features['weak_present']:
            score += self.weights['weak'] * (features['weak_count'] / 2)
        
        return max(0.0, min(100.0, score))
    
    def get_category(self, score):
        if score >= 86:
            return "Research-Level", "🔬"
        elif score >= 71:
            return "Strong", "💪"
        elif score >= 41:
            return "Moderate", "📊"
        else:
            return "Weak", "⚠️"
    
    def detect_fallacy(self, sentence):
        sentence_lower = sentence.lower()
        words = set(sentence_lower.split())
        fallacies = []
        
        if words & self.ad_hominem_insults:
            fallacies.append("Ad Hominem")
        if any(re.search(rf'\b{word}\b', sentence_lower) for word in self.generalization_indicators):
            fallacies.append("Hasty Generalization")
        if any(phrase in sentence_lower for phrase in self.slippery_slope_phrases):
            fallacies.append("Slippery Slope")
        if 'either' in sentence_lower and 'or' in sentence_lower:
            fallacies.append("False Dilemma")
        if words & self.emotional_words:
            fallacies.append("Appeal to Emotion")
        for pattern in self.circular_patterns:
            if re.search(pattern, sentence_lower):
                fallacies.append("Circular Reasoning")
                break
        
        return list(set(fallacies)) if fallacies else ["None Detected"]
    
    def determine_argument_role(self, sentence, prev_role=None):
        sentence_lower = sentence.lower()
        for role, pattern in self.role_patterns.items():
            if re.search(pattern, sentence_lower):
                return role.title()
        return "Claim" if prev_role is None else "Evidence"
    
    def generate_rewrite(self, sentence, features):
        if features['weak_present']:
            words = sentence.split()
            filtered = [w for w in words if re.sub(r'[^\w]', '', w.lower()) not in self.weak_words]
            stronger = ' '.join(filtered) if filtered else sentence
        else:
            stronger = "Already strong enough"
        
        if not features['evidence_present'] and len(sentence.split()) > 3:
            evidence_based = f"According to research, {sentence}"
        else:
            evidence_based = sentence
        
        if features['emotional_present'] or features['bias_present']:
            words = sentence.split()
            filtered = []
            for w in words:
                clean = re.sub(r'[^\w]', '', w.lower())
                if clean not in self.emotional_words and clean not in self.bias_words:
                    filtered.append(w)
            neutral = ' '.join(filtered) if filtered else sentence
        else:
            neutral = "Already neutral"
        
        return {
            'original': sentence,
            'stronger': stronger,
            'evidence_based': evidence_based,
            'neutral': neutral
        }
    
    def analyze_sentence(self, sentence, prev_role=None):
        features = self.extract_features(sentence)
        score = self.calculate_score(features)
        category, emoji = self.get_category(score)
        fallacies = self.detect_fallacy(sentence)
        argument_role = self.determine_argument_role(sentence, prev_role)
        
        return {
            'sentence': sentence,
            'score': round(score, 1),
            'category': category,
            'category_emoji': emoji,
            'detected_features': features,
            'fallacies': fallacies,
            'argument_role': argument_role
        }

# ---------------------------
# CACHED READABILITY FUNCTIONS
# ---------------------------
@st.cache_data(ttl=3600)
def get_readability_metrics(text):
    return {
        'flesch_score': textstat.flesch_reading_ease(text),
        'avg_sentence_length': textstat.avg_sentence_length(text),
        'complex_words': textstat.difficult_words(text),
        'grade_level': textstat.flesch_kincaid_grade(text),
        'complex_percentage': (textstat.difficult_words(text) / max(textstat.lexicon_count(text, True), 1)) * 100
    }

def get_readability_level(score):
    return "Easy" if score >= 60 else "Moderate" if score >= 30 else "Complex"

# ---------------------------
# CACHED FULL ANALYSIS
# ---------------------------
@st.cache_data(ttl=300)
def analyze_full_text(text, academic_mode):
    core = CoreAnalyzer()
    legal = LegalAnalyzer() if academic_mode else None
    toxicity_analyzer = ToxicityAnalyzer()
    
    sentences = sent_tokenize(text)
    
    # Core analysis
    core_results = []
    prev_role = None
    for sent in sentences:
        analysis = core.analyze_sentence(sent, prev_role)
        core_results.append(analysis)
        prev_role = analysis['argument_role']
    
    df = pd.DataFrame(core_results)
    
    # Legal analysis if enabled
    legal_results = None
    if academic_mode:
        irac_structure, missing_irac = legal.detect_irac_structure(sentences)
        formality_score = legal.calculate_formality_score(text)
        legal_flags = legal.check_legal_quality(sentences)
        citation_suggestions = legal.suggest_citation(sentences)
        coherence_score = legal.calculate_coherence_score(sentences)
        
        legal_results = {
            'irac': irac_structure,
            'missing': missing_irac,
            'formality': formality_score,
            'flags': legal_flags,
            'citations': citation_suggestions,
            'coherence': coherence_score
        }
    
    # Toxicity analysis
    toxicity_results = toxicity_analyzer.analyze_toxicity(text)
    
    # Readability
    readability = get_readability_metrics(text)
    
    return {
        'sentences': sentences,
        'df': df,
        'overall_score': df['score'].mean(),
        'overall_category': core.get_category(df['score'].mean())[0],
        'overall_emoji': core.get_category(df['score'].mean())[1],
        'legal': legal_results,
        'toxicity': toxicity_results,
        'toxicity_analyzer': toxicity_analyzer,
        'readability': readability,
        'core': core
    }

# ---------------------------
# UI COMPONENTS
# ---------------------------
def display_metric_card(title, value, subtitle=None, color="#6366F1"):
    st.markdown(f"""
    <div style="background-color: {'#F5F5F5' if st.session_state.get('academic_mode', False) else '#111827'}; 
                padding: 20px; border-radius: 12px; border-left: 4px solid {color}; margin-bottom: 10px;">
        <p style="color: {'#111111' if st.session_state.get('academic_mode', False) else '#CBD5E1'}; margin: 0; font-size: 14px;">{title}</p>
        <p style="color: {'#111111' if st.session_state.get('academic_mode', False) else '#F8FAFC'}; margin: 0; font-size: 28px; font-weight: 700;">{value}</p>
        {f'<p style="color: #22C55E; margin: 0; font-size: 14px;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def display_badge(text, category):
    color_map = {
        'Weak': '#EF4444', 'Moderate': '#F59E0B', 'Strong': '#22C55E',
        'Research-Level': '#2C3E50' if st.session_state.get('academic_mode', False) else '#3B82F6',
        'Claim': '#6366F1', 'Evidence': '#22C55E', 'Counterargument': '#F59E0B', 'Conclusion': '#3B82F6',
        'issue': '#2C3E50', 'rule': '#22C55E', 'application': '#F59E0B', 'conclusion': '#3B82F6'
    }
    color = color_map.get(category.lower() if isinstance(category, str) else category, '#6366F1')
    st.markdown(f"""
    <span style="background-color: {color}; color: white; padding: 4px 12px; 
                  border-radius: 20px; font-size: 12px; font-weight: 600; margin: 2px;">
        {text}
    </span>
    """, unsafe_allow_html=True)

def highlight_text(sentence, core):
    words = sentence.split()
    highlighted = []
    for word in words:
        clean = re.sub(r'[^\w]', '', word.lower())
        if clean in core.evidence_words:
            highlighted.append(f'<span class="highlight-evidence">{word}</span>')
        elif clean in core.logic_words:
            highlighted.append(f'<span class="highlight-logic">{word}</span>')
        elif clean in core.bias_words:
            highlighted.append(f'<span class="highlight-bias">{word}</span>')
        elif clean in core.weak_words:
            highlighted.append(f'<span style="background-color: #F97316; color: white; padding: 2px 5px; border-radius: 4px;">{word}</span>')
        elif clean in core.emotional_words:
            highlighted.append(f'<span style="background-color: #A855F7; color: white; padding: 2px 5px; border-radius: 4px;">{word}</span>')
        else:
            highlighted.append(word)
    return ' '.join(highlighted)

# ---------------------------
# MAIN APP
# ---------------------------
def main():
    # Initialize session state
    if 'academic_mode' not in st.session_state:
        st.session_state.academic_mode = False
    
    # Sidebar for mode toggle
    with st.sidebar:
        st.markdown("## ⚖️ System Mode")
        academic_mode = st.toggle(
            "🎓 Academic Legal Mode",
            value=st.session_state.academic_mode,
            help="Enable LLB-friendly legal analysis features"
        )
        st.session_state.academic_mode = academic_mode
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **Argument Strength Analyzer Pro**  
        Research-grade AI system for legal and academic debate analysis.
        
        Features:
        - Multi-factor scoring
        - Fallacy detection
        - IRAC structure (Legal Mode)
        - Toxicity analysis
        - Coherence metrics
        """)
    
    # Apply theme
    st.markdown(get_theme_css(st.session_state.academic_mode), unsafe_allow_html=True)
    
    # Header
    title = "⚖️ Argument Strength Analyzer Pro" if st.session_state.academic_mode else "🎯 Argument Strength Analyzer Pro"
    subtitle = "Legal Academic Intelligence System" if st.session_state.academic_mode else "Research-Grade Debate Intelligence System"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 30px 0;">
        <h1 style="font-size: 48px; margin-bottom: 10px;">{title}</h1>
        <p style="color: {'#111111' if st.session_state.academic_mode else '#CBD5E1'}; font-size: 18px;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzers
    core = CoreAnalyzer()
    
    # Input section
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        user_input = st.text_area(
            "Enter your argument or legal text:",
            height=200,
            placeholder="Paste your argument here... The system will analyze each sentence for strength, fallacies, structure, and toxicity.",
            key="input_text"
        )
        
        analyze_btn = st.button("🔍 Analyze Argument", use_container_width=True)
    
    if analyze_btn and user_input.strip():
        with st.spinner("Analyzing argument with advanced AI..."):
            # Run cached analysis
            results = analyze_full_text(user_input, st.session_state.academic_mode)
            
            df = results['df']
            sentences = results['sentences']
            overall_score = results['overall_score']
            overall_category = results['overall_category']
            overall_emoji = results['overall_emoji']
            readability = results['readability']
            toxicity = results['toxicity']
            toxicity_analyzer = results['toxicity_analyzer']
            
            # Build tabs based on mode
            tabs = ["📊 Overview", "📝 Sentence Analysis", "⚠️ Fallacy Report"]
            if st.session_state.academic_mode:
                tabs.extend(["🏛️ IRAC & Legal", "📚 Readability & Formality"])
            else:
                tabs.extend(["🏗️ Structure Analysis", "📖 Readability"])
            tabs.extend(["🚨 Toxicity & Ethics", "🧠 Coherence & Research"])
            
            tab_objects = st.tabs(tabs)
            
            # Tab 1: Overview
            with tab_objects[0]:
                st.markdown("### Overall Argument Strength")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    color = '#EF4444' if overall_score < 40 else '#F59E0B' if overall_score < 70 else '#22C55E' if overall_score < 86 else ('#2C3E50' if st.session_state.academic_mode else '#3B82F6')
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
                
                if st.session_state.academic_mode and results['legal']:
                    st.markdown("### Legal Quality Indicators")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        display_metric_card("Formality Score", f"{results['legal']['formality']:.0f}/100", "Academic tone")
                    with col2:
                        display_metric_card("Coherence Score", f"{results['legal']['coherence']:.0f}/100", "Logical flow")
                    with col3:
                        missing = len(results['legal']['missing'])
                        display_metric_card("IRAC Completeness", f"{4-missing}/4", f"Missing: {', '.join(results['legal']['missing']) if missing else 'None'}")
                
                st.markdown("### Toxicity Level")
                col1, col2 = st.columns([1, 3])
                with col1:
                    display_metric_card("Toxicity Score", f"{toxicity['score']:.0f}/100", toxicity['category'], toxicity['color'])
                with col2:
                    st.markdown(f"""
                    <div style="background-color: {'#F5F5F5' if st.session_state.academic_mode else '#1E293B'}; 
                                padding: 15px; border-radius: 10px; border-left: 4px solid {toxicity['color']};">
                        <p style="color: {'#111111' if st.session_state.academic_mode else '#F8FAFC'}; margin: 0;">
                            <strong>Category:</strong> {toxicity['category']}<br>
                            <strong>Shouting words:</strong> {toxicity['shouting_count']}<br>
                            {f'<strong>Suggestion:</strong> {toxicity["suggestion"]}' if toxicity['suggestion'] else ''}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        x=[f"S{i+1}" for i in range(len(df))],
                        y=df['score'],
                        color=df['category'],
                        color_discrete_map={
                            'Weak': '#EF4444', 'Moderate': '#F59E0B',
                            'Strong': '#22C55E', 'Research-Level': '#2C3E50' if st.session_state.academic_mode else '#3B82F6'
                        },
                        title="Sentence Strength Scores"
                    )
                    fig.update_layout(
                        plot_bgcolor='#F5F5F5' if st.session_state.academic_mode else '#1E293B',
                        paper_bgcolor='#F5F5F5' if st.session_state.academic_mode else '#1E293B',
                        font_color='#111111' if st.session_state.academic_mode else '#F8FAFC'
                    )
                    st.plotly_chart(fig, width="stretch")
                
                with col2:
                    category_counts = df['category'].value_counts()
                    fig = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title="Argument Distribution",
                        color_discrete_map={
                            'Weak': '#EF4444', 'Moderate': '#F59E0B',
                            'Strong': '#22C55E', 'Research-Level': '#2C3E50' if st.session_state.academic_mode else '#3B82F6'
                        }
                    )
                    fig.update_layout(
                        plot_bgcolor='#F5F5F5' if st.session_state.academic_mode else '#1E293B',
                        paper_bgcolor='#F5F5F5' if st.session_state.academic_mode else '#1E293B',
                        font_color='#111111' if st.session_state.academic_mode else '#F8FAFC'
                    )
                    st.plotly_chart(fig, width="stretch")
                
                # Radar chart with bounded values
                st.markdown("### Multidimensional Analysis")
                
                logic_score = min(100, (df['detected_features'].apply(lambda x: x['logic_count']).mean() / 3) * 100)
                evidence_score = min(100, (df['detected_features'].apply(lambda x: x['evidence_count']).mean() / 3) * 100)
                bias_score = min(100, max(0, 100 - ((df['detected_features'].apply(lambda x: x['bias_count']).mean() / 3) * 100)))
                clarity_score = min(100, df.apply(lambda x: 90 if get_readability_level(textstat.flesch_reading_ease(x['sentence'])) == 'Easy' 
                                                  else 60 if get_readability_level(textstat.flesch_reading_ease(x['sentence'])) == 'Moderate' 
                                                  else 30, axis=1).mean())
                
                if st.session_state.academic_mode and results['legal']:
                    formality_radar = results['legal']['formality']
                    coherence_radar = results['legal']['coherence']
                    toxicity_radar = 100 - toxicity['score']
                    
                    fig = go.Figure(data=go.Scatterpolar(
                        r=[logic_score, evidence_score, bias_score, clarity_score, formality_radar, coherence_radar, toxicity_radar],
                        theta=['Logic', 'Evidence', 'Low Bias', 'Clarity', 'Formality', 'Coherence', 'Low Toxicity'],
                        fill='toself',
                        line=dict(color='#2C3E50' if st.session_state.academic_mode else '#6366F1', width=3)
                    ))
                else:
                    emotional_score = min(100, max(0, 100 - ((df['detected_features'].apply(lambda x: x['emotional_count']).mean() / 3) * 100)))
                    fig = go.Figure(data=go.Scatterpolar(
                        r=[logic_score, evidence_score, bias_score, clarity_score, emotional_score],
                        theta=['Logic', 'Evidence', 'Low Bias', 'Clarity', 'Low Emotion'],
                        fill='toself',
                        line=dict(color='#6366F1', width=3)
                    ))
                
                fig.update_layout(
                    polar=dict(
                        bgcolor='#F5F5F5' if st.session_state.academic_mode else '#1E293B',
                        radialaxis=dict(visible=True, range=[0, 100], 
                                       color='#111111' if st.session_state.academic_mode else '#CBD5E1')
                    ),
                    showlegend=False,
                    paper_bgcolor='#F5F5F5' if st.session_state.academic_mode else '#1E293B',
                    font_color='#111111' if st.session_state.academic_mode else '#F8FAFC'
                )
                st.plotly_chart(fig, width="stretch")
            
            # Tab 2: Sentence Analysis
            with tab_objects[1]:
                st.markdown("### Detailed Sentence Analysis")
                
                for idx, row in df.iterrows():
                    with st.container():
                        st.markdown(f"#### Sentence {idx + 1}")
                        
                        # Apply toxicity highlighting if needed
                        if any(word.lower() in row['sentence'].lower() for word in toxicity['abusive_words']):
                            highlighted = toxicity_analyzer.highlight_toxic_text(row['sentence'], toxicity['abusive_words'])
                        else:
                            highlighted = highlight_text(row['sentence'], core)
                        
                        st.markdown(f"<div style='background-color: {'#F5F5F5' if st.session_state.academic_mode else '#111827'}; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>{highlighted}</div>", unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            display_badge(f"{row['category']} {row['category_emoji']}", row['category'])
                        with col2:
                            display_badge(f"Score: {row['score']:.0f}", row['category'])
                        with col3:
                            display_badge(row['argument_role'], row['argument_role'])
                        with col4:
                            readability_score = textstat.flesch_reading_ease(row['sentence'])
                            display_badge(get_readability_level(readability_score), 'Moderate')
                        
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
                                rewrites = core.generate_rewrite(row['sentence'], features)
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Original:**")
                                    st.markdown(f"<div style='background-color: {'#F5F5F5' if st.session_state.academic_mode else '#1E293B'}; padding: 10px; border-radius: 5px;'>{rewrites['original']}</div>", unsafe_allow_html=True)
                                    st.markdown("**Stronger Version:**")
                                    st.markdown(f"<div style='background-color: {'#F5F5F5' if st.session_state.academic_mode else '#1E293B'}; padding: 10px; border-radius: 5px; border-left: 3px solid #22C55E;'>{rewrites['stronger']}</div>", unsafe_allow_html=True)
                                with col2:
                                    st.markdown("**Evidence-Based:**")
                                    st.markdown(f"<div style='background-color: {'#F5F5F5' if st.session_state.academic_mode else '#1E293B'}; padding: 10px; border-radius: 5px; border-left: 3px solid #3B82F6;'>{rewrites['evidence_based']}</div>", unsafe_allow_html=True)
                                    st.markdown("**Neutral Version:**")
                                    st.markdown(f"<div style='background-color: {'#F5F5F5' if st.session_state.academic_mode else '#1E293B'}; padding: 10px; border-radius: 5px; border-left: 3px solid #F59E0B;'>{rewrites['neutral']}</div>", unsafe_allow_html=True)
                        
                        st.markdown("---")
            
            # Tab 3: Fallacy Report
            with tab_objects[2]:
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
                        plot_bgcolor='#F5F5F5' if st.session_state.academic_mode else '#1E293B',
                        paper_bgcolor='#F5F5F5' if st.session_state.academic_mode else '#1E293B',
                        font_color='#111111' if st.session_state.academic_mode else '#F8FAFC'
                    )
                    st.plotly_chart(fig, width="stretch")
            
            # Tab 4 depends on mode
            tab4_idx = 3
            
            if st.session_state.academic_mode:
                # Tab 4: IRAC & Legal
                with tab_objects[tab4_idx]:
                    st.markdown("### IRAC Legal Structure Analysis")
                    
                    if results['legal']:
                        # IRAC breakdown
                        st.markdown("#### Sentence Classification")
                        irac_df = pd.DataFrame([
                            {
                                'Sentence': f"S{i+1}: {item['sentence'][:80]}...",
                                'IRAC Role': item['role'].title(),
                                'Confidence': f"{item['confidence']*100:.0f}%"
                            }
                            for i, item in enumerate(results['legal']['irac'])
                        ])
                        st.dataframe(irac_df, use_container_width=True)
                        
                        # Structure completeness
                        st.markdown("#### Structure Completeness")
                        missing = results['legal']['missing']
                        if missing:
                            st.warning(f"⚠️ Missing IRAC components: {', '.join(missing).title()}")
                        else:
                            st.success("✅ Complete IRAC structure detected!")
                        
                        # Formality and legal quality
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### Formality Score")
                            formality = results['legal']['formality']
                            color = '#22C55E' if formality >= 70 else '#F59E0B' if formality >= 50 else '#EF4444'
                            st.markdown(f"""
                            <div style="background-color: {'#F5F5F5' if st.session_state.academic_mode else '#111827'}; 
                                        padding: 20px; border-radius: 12px; border-left: 4px solid {color};">
                                <p style="font-size: 36px; font-weight: 700; color: {color};">{formality:.0f}/100</p>
                                <p>{'Formal academic tone' if formality >= 70 else 'Moderately formal' if formality >= 50 else 'Informal tone - revise'}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("#### Legal Quality Flags")
                            if results['legal']['flags']:
                                for flag in results['legal']['flags']:
                                    st.error(f"**{flag['issue']}**\n\n{flag['sentence'][:150]}...\n\n💡 {flag['suggestion']}")
                            else:
                                st.success("No legal quality issues detected")
                        
                        # Citation suggestions
                        st.markdown("#### Citation Suggestions")
                        if results['legal']['citations']:
                            for cit in results['legal']['citations']:
                                st.info(f"📚 {cit['suggestion']}\n\n> {cit['sentence'][:200]}...")
                        else:
                            st.success("No citation suggestions needed")
            else:
                # Tab 4: Structure Analysis (non-academic mode)
                with tab_objects[tab4_idx]:
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
                        st.plotly_chart(fig, width="stretch")
                    
                    with col2:
                        flow_data = []
                        for i, row in df.iterrows():
                            flow_data.append({'Position': i+1, 'Role': row['argument_role'], 'Score': row['score']})
                        flow_df = pd.DataFrame(flow_data)
                        fig = px.line(
                            flow_df, x='Position', y='Score', text='Role',
                            title="Argument Flow and Strength", markers=True
                        )
                        fig.update_traces(line=dict(color='#6366F1', width=3))
                        fig.update_layout(
                            plot_bgcolor='#1E293B',
                            paper_bgcolor='#1E293B',
                            font_color='#F8FAFC'
                        )
                        st.plotly_chart(fig, width="stretch")
                    
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
            
            # Tab 5 depends on mode
            tab5_idx = 4
            
            if st.session_state.academic_mode:
                # Tab 5: Readability & Formality
                with tab_objects[tab5_idx]:
                    st.markdown("### Readability & Formality Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        level_color = '#22C55E' if readability['flesch_score'] >= 60 else '#F59E0B' if readability['flesch_score'] >= 30 else '#EF4444'
                        display_metric_card("Flesch Reading Ease", f"{readability['flesch_score']:.1f}", 
                                           get_readability_level(readability['flesch_score']), level_color)
                    with col2:
                        display_metric_card("Grade Level", f"Grade {readability['grade_level']:.1f}",
                                           f"{readability['avg_sentence_length']:.0f} words/sentence")
                    with col3:
                        display_metric_card("Complex Words", f"{readability['complex_words']}",
                                           f"{readability['complex_percentage']:.1f}% of text")
                    
                    # Sentence readability
                    st.markdown("### Sentence Readability")
                    readability_df = pd.DataFrame([
                        {
                            'Sentence': f"S{i+1}",
                            'Score': textstat.flesch_reading_ease(row['sentence']),
                            'Level': get_readability_level(textstat.flesch_reading_ease(row['sentence'])),
                            'Words': row['detected_features']['word_count']
                        }
                        for i, row in df.iterrows()
                    ])
                    
                    fig = px.bar(
                        readability_df, x='Sentence', y='Score', color='Level',
                        color_discrete_map={'Easy': '#22C55E', 'Moderate': '#F59E0B', 'Complex': '#EF4444'},
                        title="Readability Scores by Sentence"
                    )
                    fig.update_layout(
                        plot_bgcolor='#F5F5F5' if st.session_state.academic_mode else '#1E293B',
                        paper_bgcolor='#F5F5F5' if st.session_state.academic_mode else '#1E293B',
                        font_color='#111111' if st.session_state.academic_mode else '#F8FAFC'
                    )
                    st.plotly_chart(fig, width="stretch")
                    
                    # Formality tips
                    if results['legal']['formality'] < 70:
                        st.warning("""
                        **Formality Improvement Tips:**
                        - Replace personal pronouns with passive constructions
                        - Use formal legal vocabulary (herein, thereof, pursuant)
                        - Avoid contractions and informal phrases
                        - Maintain objective, dispassionate tone
                        """)
            else:
                # Tab 5: Readability (non-academic mode)
                with tab_objects[tab5_idx]:
                    st.markdown("### Readability Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        level_color = '#22C55E' if readability['flesch_score'] >= 60 else '#F59E0B' if readability['flesch_score'] >= 30 else '#EF4444'
                        display_metric_card("Flesch Reading Ease", f"{readability['flesch_score']:.1f}", 
                                           get_readability_level(readability['flesch_score']), level_color)
                    with col2:
                        display_metric_card("Grade Level", f"Grade {readability['grade_level']:.1f}",
                                           f"{readability['avg_sentence_length']:.0f} words/sentence")
                    with col3:
                        display_metric_card("Complex Words", f"{readability['complex_words']}",
                                           f"{readability['complex_percentage']:.1f}% of text")
                    
                    st.markdown("### Sentence Readability Breakdown")
                    readability_df = pd.DataFrame([
                        {
                            'Sentence': f"S{i+1}",
                            'Score': textstat.flesch_reading_ease(row['sentence']),
                            'Level': get_readability_level(textstat.flesch_reading_ease(row['sentence'])),
                            'Words': row['detected_features']['word_count']
                        }
                        for i, row in df.iterrows()
                    ])
                    
                    fig = px.bar(
                        readability_df, x='Sentence', y='Score', color='Level',
                        color_discrete_map={'Easy': '#22C55E', 'Moderate': '#F59E0B', 'Complex': '#EF4444'},
                        title="Readability Scores by Sentence"
                    )
                    fig.update_layout(
                        plot_bgcolor='#1E293B',
                        paper_bgcolor='#1E293B',
                        font_color='#F8FAFC'
                    )
                    st.plotly_chart(fig, width="stretch")
            
            # Tab 6: Toxicity & Ethics
            tab6_idx = 5
            with tab_objects[tab6_idx]:
                st.markdown("### Toxicity & Ethics Analysis")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"""
                    <div style="background-color: {'#F5F5F5' if st.session_state.academic_mode else '#111827'}; 
                                padding: 20px; border-radius: 12px; border-left: 4px solid {toxicity['color']};">
                        <p style="color: {'#111111' if st.session_state.academic_mode else '#CBD5E1'}; margin: 0;">Toxicity Score</p>
                        <p style="color: {toxicity['color']}; font-size: 48px; font-weight: 700;">{toxicity['score']:.0f}</p>
                        <p style="color: {'#111111' if st.session_state.academic_mode else '#F8FAFC'};"><strong>Category:</strong> {toxicity['category']}</p>
                        <p style="color: {'#111111' if st.session_state.academic_mode else '#F8FAFC'};"><strong>Shouting words:</strong> {toxicity['shouting_count']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if toxicity['suggestion']:
                        st.info(f"💡 {toxicity['suggestion']}")
                
                with col2:
                    # Highlighted text
                    st.markdown("#### Text with Toxic Elements Highlighted")
                    highlighted_full = toxicity_analyzer.highlight_toxic_text(user_input, toxicity['abusive_words'])
                    st.markdown(f"""
                    <div style="background-color: {'#F5F5F5' if st.session_state.academic_mode else '#1E293B'}; 
                                padding: 20px; border-radius: 12px; max-height: 300px; overflow-y: auto;">
                        {highlighted_full}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Toxic matches table
                if toxicity['toxic_matches']:
                    st.markdown("#### Detected Toxic Patterns")
                    toxic_df = pd.DataFrame([
                        {'Category': cat, 'Match': match}
                        for cat, match in toxicity['toxic_matches']
                    ])
                    st.dataframe(toxic_df, use_container_width=True)
                
                # Ethics guidelines
                st.markdown("#### Ethical Communication Guidelines")
                st.markdown("""
                - **Respectful Discourse**: Address arguments, not individuals
                - **Measured Language**: Avoid absolute claims and emotional outbursts
                - **Constructive Criticism**: Focus on ideas, not personal attacks
                - **Academic Tone**: Maintain objectivity and professionalism
                """)
            
            # Tab 7: Coherence & Research
            tab7_idx = 6
            with tab_objects[tab7_idx]:
                st.markdown("### Coherence & Research Suggestions")
                
                if st.session_state.academic_mode and results['legal']:
                    col1, col2 = st.columns(2)
                    with col1:
                        coherence = results['legal']['coherence']
                        color = '#22C55E' if coherence >= 70 else '#F59E0B' if coherence >= 50 else '#EF4444'
                        st.markdown(f"""
                        <div style="background-color: {'#F5F5F5' if st.session_state.academic_mode else '#111827'}; 
                                    padding: 20px; border-radius: 12px; border-left: 4px solid {color};">
                            <p style="color: {'#111111' if st.session_state.academic_mode else '#CBD5E1'}; margin: 0;">Argument Coherence</p>
                            <p style="color: {color}; font-size: 48px; font-weight: 700;">{coherence:.0f}/100</p>
                            <p>{'Excellent logical flow' if coherence >= 70 else 'Moderate coherence - improve transitions' if coherence >= 50 else 'Weak coherence - restructure argument'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### Transition Word Check")
                        transition_counts = {}
                        text_lower = user_input.lower()
                        for cat, words in core.transition_words.items():
                            count = sum(1 for w in words if re.search(rf'\b{w}\b', text_lower))
                            transition_counts[cat] = count
                        
                        trans_df = pd.DataFrame([
                            {'Category': cat.title(), 'Count': count}
                            for cat, count in transition_counts.items()
                        ])
                        st.dataframe(trans_df, use_container_width=True)
                    
                    # Research suggestions
                    st.markdown("#### Research Enhancement Suggestions")
                    if results['legal']['citations']:
                        for cit in results['legal']['citations']:
                            st.info(f"📚 {cit['suggestion']}\n\n> {cit['sentence'][:150]}...")
                    else:
                        st.success("✅ Claims appear well-supported")
                    
                    # Evidence gaps
                    evidence_gaps = df[~df['detected_features'].apply(lambda x: x['evidence_present'])]
                    if not evidence_gaps.empty:
                        st.warning(f"⚠️ {len(evidence_gaps)} sentences lack evidentiary support")
                        for _, row in evidence_gaps.iterrows():
                            st.markdown(f"- {row['sentence'][:100]}...")
                    
                    # Bias check
                    bias_present = df[df['detected_features'].apply(lambda x: x['bias_present'])]
                    if not bias_present.empty:
                        st.warning(f"⚠️ {len(bias_present)} sentences contain absolute language")
                else:
                    st.markdown("#### Enable Academic Legal Mode for coherence analysis")
                    st.info("🎓 Toggle Academic Legal Mode in sidebar for detailed coherence metrics, IRAC structure, and research suggestions")
    
    # Footer
    st.markdown("---")
    footer_color = '#111111' if st.session_state.academic_mode else '#CBD5E1'
    st.markdown(f"""
    <div style="text-align: center; color: {footer_color}; padding: 20px;">
        <p>⚖️ Argument Strength Analyzer Pro | Legal Academic Intelligence System</p>
        <p style="font-size: 12px;">Built with Streamlit, spaCy, NLTK, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
