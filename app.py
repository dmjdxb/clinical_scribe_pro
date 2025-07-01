"""
Clinical Scribe Pro - Complete Integrated System
Includes all features: Live transcription, multilingual support, 
functional medicine, IRAC intake form, and MedNet Flow integration
"""

import streamlit as st
import speech_recognition as sr
import threading
import queue
import time
import openai
import numpy as np
from datetime import datetime
import os
import json
import re
from typing import Dict, List, Optional, Any
import sqlite3
import pandas as pd
from dataclasses import dataclass
import wave
import io
import pyperclip
import xml.etree.ElementTree as ET
from collections import defaultdict

# Page config - MUST BE FIRST
st.set_page_config(
    page_title="Clinical Scribe Pro - Dr. Shefali Verma",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful, mobile-friendly design
st.markdown("""
<style>
    /* Main container padding */
    .main {
        padding: 1rem;
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Large record button */
    .stButton > button {
        width: 100%;
        font-weight: bold;
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    
    /* Record button specific styling */
    .record-button > button {
        min-height: 80px;
        font-size: 24px;
        background-color: #FF4B4B;
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
    }
    
    .record-button > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
    }
    
    /* Generate button styling */
    .generate-button > button {
        min-height: 60px;
        font-size: 20px;
        background-color: #0891B2;
        color: white;
        border: none;
    }
    
    /* Language pills */
    .language-pill {
        display: inline-block;
        padding: 8px 16px;
        margin: 4px;
        border-radius: 20px;
        background-color: #F3F4F6;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .language-pill:hover {
        background-color: #E5E7EB;
        transform: translateY(-1px);
    }
    
    .language-pill.selected {
        background-color: #0891B2;
        color: white;
    }
    
    /* Live transcription box */
    .transcription-box {
        background-color: #F8F9FA;
        border: 2px solid #E5E7EB;
        border-radius: 15px;
        padding: 20px;
        min-height: 200px;
        font-size: 16px;
        line-height: 1.8;
        margin: 20px 0;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Medical term highlighting */
    .medical-term {
        background-color: #FEF3C7;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 500;
    }
    
    /* Pain scale emoji */
    .pain-emoji {
        font-size: 48px;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem;
        }
        .stButton > button {
            min-height: 60px;
            font-size: 18px;
        }
    }
    
    /* Pulse animation for recording */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .recording-indicator {
        animation: pulse 1.5s infinite;
        color: #FF4B4B;
        font-size: 24px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'clinical_note' not in st.session_state:
    st.session_state.clinical_note = ""
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "Auto-detect"
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = ""
if 'consultation_type' not in st.session_state:
    st.session_state.consultation_type = "General"
if 'intake_form' not in st.session_state:
    st.session_state.intake_form = None
if 'show_intake_form' not in st.session_state:
    st.session_state.show_intake_form = False

# ============================================
# DATA STRUCTURES AND CONSTANTS
# ============================================

# Medical terminology for highlighting
MEDICAL_TERMS = {
    "symptoms": ["pain", "fever", "cough", "headache", "nausea", "fatigue", "dizziness", 
                 "ألم", "حمى", "سعال", "صداع", "غثيان", "إرهاق", "دوخة",
                 "douleur", "fièvre", "toux", "mal de tête", "nausée"],
    "conditions": ["diabetes", "hypertension", "asthma", "السكري", "ضغط الدم", "الربو"],
    "medications": ["paracetamol", "ibuprofen", "antibiotic", "باراسيتامول", "مضاد حيوي"],
    "time": ["days", "weeks", "months", "يوم", "أسبوع", "شهر", "jours", "semaines"]
}

# Dialect detection patterns
DIALECT_PATTERNS = {
    'ar_gulf': ['الحين', 'دحين', 'يعورني', 'وايد'],
    'ar_egyptian': ['دلوقتي', 'إزاي', 'النهاردة', 'إمبارح'],
    'ar_levantine': ['هلأ', 'كيف', 'شو', 'بدي'],
    'fr': ['bonjour', 'j\'ai', 'mal', 'depuis']
}

# Functional medicine terms
FUNCTIONAL_MEDICINE_TERMS = {
    "gut_health": ["microbiome", "dysbiosis", "leaky gut", "SIBO", "candida"],
    "hormones": ["adrenal", "thyroid", "cortisol", "estrogen", "testosterone"],
    "supplements": ["probiotic", "vitamin D", "magnesium", "omega-3", "glutamine"]
}

# ============================================
# IRAC INTAKE FORM DATA STRUCTURE
# ============================================

@dataclass
class IRACIntakeForm:
    """Structure matching Dr. Shefali's IRAC intake form"""
    
    # Demographics
    name: str = ""
    date_of_birth: str = ""
    age: str = ""
    gender: str = ""
    blood_group: str = ""
    marital_status: str = ""
    children: str = ""
    
    # Measurements
    height: str = ""
    weight: str = ""
    blood_pressure: str = ""
    pulse_rate: str = ""
    temperature: str = ""
    
    # Pain Assessment
    pain_location: str = ""
    pain_duration: str = ""
    pain_frequency: str = ""
    pain_scale: int = 0
    pain_notes: str = ""
    pain_goal: str = ""
    
    # Observation
    nails_observation: str = ""
    tongue_observation: str = ""
    eyes_observation: str = ""
    hairline_observation: str = ""
    
    # History
    history_notes: str = ""
    work_schedule: str = ""
    eating_habits: str = ""
    hydration: str = ""
    sleeping: str = ""
    
    # Bowel Habits/Digestive
    bowel_habits: str = ""
    digestive_symptoms: str = ""
    
    # Common Conditions
    asthma: str = ""
    eczema: str = ""
    hayfever: str = ""
    migraines: str = ""
    
    # Premenstrual Symptoms
    menstrual_regular: str = ""
    menstrual_cycle: str = ""
    pms: str = ""
    breast_tenderness: str = ""
    menstrual_flow: str = ""
    pregnancy_history: str = ""
    easy_to_conceive: str = ""
    
    # Mental Clarity
    mental_clarity: str = ""
    
    # Medical History
    past_medical_history: str = ""
    hospital_admissions: str = ""
    surgeries: str = ""
    medications_supplements: str = ""
    family_history: str = ""
    
    # Social History
    social_history: str = ""
    allergies: str = ""
    travel_history: str = ""
    stress_level: str = ""
    energy_level: str = ""
    libido: str = ""
    exercise_history: str = ""
    
    # Recommendations
    recommendations: str = ""

# ============================================
# DATABASE SETUP
# ============================================

def init_database():
    """Initialize SQLite database for storing consultations"""
    conn = sqlite3.connect('clinical_scribe.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS consultations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  patient_name TEXT,
                  language TEXT,
                  dialect TEXT,
                  consultation_type TEXT,
                  transcription TEXT,
                  clinical_note TEXT,
                  intake_form TEXT,
                  audio_path TEXT)''')
    
    conn.commit()
    conn.close()

# Initialize database
init_database()

# ============================================
# ENHANCED MEDICAL TRANSCRIBER CLASS
# ============================================

class MedicalTranscriber:
    """Enhanced medical transcriber with multilingual and functional medicine support"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_recording = False
        self.text_queue = queue.Queue()
        self.audio_data = []
        
    def detect_dialect(self, text):
        """Detect Arabic dialect from text"""
        for dialect, patterns in DIALECT_PATTERNS.items():
            if any(pattern in text for pattern in patterns):
                return dialect
        return None
        
    def highlight_medical_terms(self, text):
        """Highlight medical terms in text"""
        highlighted = text
        
        # Highlight standard medical terms
        for category, terms in MEDICAL_TERMS.items():
            for term in terms:
                if term.lower() in highlighted.lower():
                    highlighted = highlighted.replace(term, f'<span class="medical-term">{term}</span>')
        
        # Highlight functional medicine terms
        for category, terms in FUNCTIONAL_MEDICINE_TERMS.items():
            for term in terms:
                if term.lower() in highlighted.lower():
                    highlighted = highlighted.replace(term, f'<span class="medical-term">{term}</span>')
        
        return highlighted
        
    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
        
    def stop_recording(self):
        self.is_recording = False
        return self.audio_data
        
    def _record_audio(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
        while self.is_recording:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    self.audio_data.append(audio)
                    
                try:
                    text = self.recognizer.recognize_google(audio)
                    self.text_queue.put(text)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    self.text_queue.put(f"Error: {str(e)}")
                    
            except sr.WaitTimeoutError:
                pass

# ============================================
# INTAKE FORM EXTRACTOR
# ============================================

class IntakeFormExtractor:
    """Extracts intake form data from consultation transcripts"""
    
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
        
    def extract_form_data(self, transcript: str, language: str = "en") -> IRACIntakeForm:
        """Extract intake form data from transcript using GPT-4"""
        
        prompt = f"""
        Extract patient intake information from this medical consultation transcript.
        This is for Dr. Shefali Verma's functional medicine practice.
        
        Transcript:
        {transcript}
        
        Extract all available information for these categories:
        - Demographics (name, age, gender, etc.)
        - Vital signs and measurements
        - Pain assessment (location, duration, scale 0-10)
        - Current symptoms and complaints
        - Digestive health and bowel habits
        - Sleep patterns and energy levels
        - Stress levels
        - Current medications and supplements
        - Allergies
        - Past medical history
        - Exercise habits
        - For female patients: menstrual history
        
        Return as JSON with all fields, use empty string for missing data.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a medical assistant extracting patient intake information for a functional medicine practice."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            extracted_data = json.loads(response.choices[0].message.content)
            return self._map_to_intake_form(extracted_data)
            
        except Exception as e:
            st.error(f"Error extracting form data: {str(e)}")
            return IRACIntakeForm()
    
    def _map_to_intake_form(self, data: Dict) -> IRACIntakeForm:
        """Map extracted data to IRACIntakeForm structure"""
        form = IRACIntakeForm()
        
        # Map all fields from JSON to form
        for key, value in data.items():
            if hasattr(form, key):
                setattr(form, key, value)
        
        return form

# ============================================
# MEDNET FLOW INTEGRATION
# ============================================

class MedNetFlowIntegration:
    """Integration handler for MedNet Flow EMR system"""
    
    def __init__(self):
        self.export_formats = ["Clipboard", "XML", "Direct Entry Guide", "Section by Section"]
        
    def prepare_for_mednet(self, clinical_note: str, patient_data: Dict, 
                          intake_form: Optional[IRACIntakeForm] = None) -> Dict:
        """Prepare clinical data in MedNet-compatible format"""
        
        sections = self._parse_soap_note(clinical_note)
        
        # Integrate intake form data if available
        if intake_form:
            patient_data.update({
                'height': intake_form.height,
                'weight': intake_form.weight,
                'blood_pressure': intake_form.blood_pressure,
                'pulse_rate': intake_form.pulse_rate,
                'temperature': intake_form.temperature,
                'allergies': intake_form.allergies,
                'medications': intake_form.medications_supplements
            })
        
        mednet_data = {
            'patient': {
                'name': patient_data.get('patient_name', ''),
                'dob': patient_data.get('date_of_birth', ''),
                'gender': patient_data.get('gender', ''),
                'blood_group': patient_data.get('blood_group', '')
            },
            'consultation': {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': datetime.now().strftime('%H:%M'),
                'type': patient_data.get('consultation_type', 'General'),
                'provider': 'Dr. Shefali Verma'
            },
            'vitals': {
                'blood_pressure': patient_data.get('blood_pressure', ''),
                'pulse': patient_data.get('pulse_rate', ''),
                'temperature': patient_data.get('temperature', ''),
                'height': patient_data.get('height', ''),
                'weight': patient_data.get('weight', '')
            },
            'clinical_notes': {
                'chief_complaint': sections.get('subjective', {}).get('chief_complaint', ''),
                'history': sections.get('subjective', {}).get('history', ''),
                'examination': sections.get('objective', ''),
                'diagnosis': sections.get('assessment', ''),
                'treatment_plan': sections.get('plan', ''),
                'full_note': clinical_note
            },
            'allergies': patient_data.get('allergies', ''),
            'medications': self._extract_medications(sections.get('plan', ''))
        }
        
        return mednet_data
    
    def _parse_soap_note(self, note: str) -> Dict:
        """Parse SOAP note into structured sections"""
        sections = {
            'subjective': {},
            'objective': '',
            'assessment': '',
            'plan': ''
        }
        
        # Split by sections
        current_section = None
        content = []
        
        for line in note.split('\n'):
            line = line.strip()
            if 'SUBJECTIVE' in line.upper():
                current_section = 'subjective'
                content = []
            elif 'OBJECTIVE' in line.upper():
                if current_section == 'subjective':
                    sections['subjective'] = {'history': '\n'.join(content)}
                current_section = 'objective'
                content = []
            elif 'ASSESSMENT' in line.upper():
                if current_section == 'objective':
                    sections['objective'] = '\n'.join(content)
                current_section = 'assessment'
                content = []
            elif 'PLAN' in line.upper():
                if current_section == 'assessment':
                    sections['assessment'] = '\n'.join(content)
                current_section = 'plan'
                content = []
            elif current_section and line:
                content.append(line)
        
        if current_section == 'plan':
            sections['plan'] = '\n'.join(content)
        
        # Extract chief complaint from subjective
        if sections['subjective']:
            lines = sections['subjective'].get('history', '').split('\n')
            if lines:
                sections['subjective']['chief_complaint'] = lines[0]
        
        return sections
    
    def _extract_medications(self, plan_text: str) -> List[Dict]:
        """Extract medications from plan section"""
        medications = []
        
        med_keywords = ['prescribe', 'start', 'continue', 'medication', 'supplement', 'mg', 'daily']
        
        lines = plan_text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in med_keywords):
                medications.append({
                    'description': line.strip('- ').strip()
                })
        
        return medications
    
    def generate_clipboard_format(self, mednet_data: Dict) -> str:
        """Generate formatted text for MedNet copy-paste"""
        
        clipboard_text = f"""=== PATIENT INFORMATION ===
Name: {mednet_data['patient']['name']}
Date: {mednet_data['consultation']['date']}
Time: {mednet_data['consultation']['time']}
Provider: {mednet_data['consultation']['provider']}

=== VITAL SIGNS ===
BP: {mednet_data['vitals']['blood_pressure']}
Pulse: {mednet_data['vitals']['pulse']}
Temp: {mednet_data['vitals']['temperature']}
Height: {mednet_data['vitals']['height']}
Weight: {mednet_data['vitals']['weight']}

=== CHIEF COMPLAINT ===
{mednet_data['clinical_notes']['chief_complaint']}

=== HISTORY ===
{mednet_data['clinical_notes']['history']}

=== EXAMINATION ===
{mednet_data['clinical_notes']['examination']}

=== ASSESSMENT/DIAGNOSIS ===
{mednet_data['clinical_notes']['diagnosis']}

=== TREATMENT PLAN ===
{mednet_data['clinical_notes']['treatment_plan']}

=== ALLERGIES ===
{mednet_data['allergies'] or 'NKDA'}

=== MEDICATIONS/SUPPLEMENTS ==="""
        
        for med in mednet_data['medications']:
            clipboard_text += f"\n- {med['description']}"
        
        return clipboard_text
    
    def generate_entry_guide(self, mednet_data: Dict) -> str:
        """Generate step-by-step guide for MedNet entry"""
        
        guide = f"""📋 MEDNET FLOW ENTRY GUIDE
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

STEP 1: PATIENT SEARCH
• Go to flow.mednetlabs.com
• Search: {mednet_data['patient']['name']}
• Or create new patient

STEP 2: START CONSULTATION
• Click "New Consultation"
• Type: {mednet_data['consultation']['type']}
• Date: Today

STEP 3: VITAL SIGNS TAB
• BP: {mednet_data['vitals']['blood_pressure']}
• Pulse: {mednet_data['vitals']['pulse']}
• Temp: {mednet_data['vitals']['temperature']}
• Height: {mednet_data['vitals']['height']}
• Weight: {mednet_data['vitals']['weight']}

STEP 4: CHIEF COMPLAINT TAB
Copy and paste:
{mednet_data['clinical_notes']['chief_complaint']}

STEP 5: HISTORY TAB
Copy and paste:
{mednet_data['clinical_notes']['history']}

STEP 6: EXAMINATION TAB
Copy and paste:
{mednet_data['clinical_notes']['examination']}

STEP 7: ASSESSMENT TAB
Copy and paste:
{mednet_data['clinical_notes']['diagnosis']}

STEP 8: PLAN TAB
Copy and paste:
{mednet_data['clinical_notes']['treatment_plan']}

STEP 9: MEDICATIONS TAB"""
        
        for i, med in enumerate(mednet_data['medications'], 1):
            guide += f"\n{i}. {med['description']}"
        
        guide += f"\n\nSTEP 10: SAVE\n• Review all tabs\n• Click 'Save Consultation'"
        
        return guide

# ============================================
# UI COMPONENTS
# ============================================

def render_intake_form_ui(form: IRACIntakeForm, missing_info: List[str]):
    """Render the IRAC intake form for review"""
    
    st.markdown("### 📋 IRAC Intake Form Review")
    
    if missing_info:
        st.warning(f"⚠️ Missing: {', '.join(missing_info[:5])}")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Demographics", "Vitals & Pain", "History", "Lifestyle", "Medical"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            form.name = st.text_input("Name", form.name)
            form.age = st.text_input("Age", form.age)
            form.gender = st.selectbox("Gender", ["", "Male", "Female"], 
                                      index=0 if not form.gender else ["", "Male", "Female"].index(form.gender))
        with col2:
            form.date_of_birth = st.text_input("DOB", form.date_of_birth)
            form.blood_group = st.text_input("Blood Group", form.blood_group)
            form.marital_status = st.text_input("Marital Status", form.marital_status)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            form.blood_pressure = st.text_input("BP", form.blood_pressure)
            form.pulse_rate = st.text_input("Pulse", form.pulse_rate)
            form.temperature = st.text_input("Temp", form.temperature)
        with col2:
            form.height = st.text_input("Height", form.height)
            form.weight = st.text_input("Weight", form.weight)
        
        st.subheader("Pain Assessment")
        form.pain_location = st.text_input("Location", form.pain_location)
        col1, col2, col3 = st.columns(3)
        with col1:
            form.pain_duration = st.text_input("Duration", form.pain_duration)
        with col2:
            form.pain_frequency = st.text_input("Frequency", form.pain_frequency)
        with col3:
            form.pain_scale = st.slider("Scale", 0, 10, form.pain_scale)
        
        # Pain emoji
        pain_emojis = ["😊", "😐", "😕", "😣", "😖", "😭"]
        emoji_index = min(int(form.pain_scale / 2), 5)
        st.markdown(f'<div class="pain-emoji">{pain_emojis[emoji_index]}</div>', unsafe_allow_html=True)
    
    with tab3:
        form.eating_habits = st.text_area("Eating Habits", form.eating_habits, height=100)
        form.sleeping = st.text_area("Sleep Patterns", form.sleeping, height=100)
        form.bowel_habits = st.text_area("Bowel Habits", form.bowel_habits, height=100)
        form.stress_level = st.text_area("Stress Level", form.stress_level, height=100)
    
    with tab4:
        form.exercise_history = st.text_area("Exercise", form.exercise_history, height=100)
        form.energy_level = st.text_area("Energy Level", form.energy_level, height=100)
        
        if form.gender == "Female":
            st.subheader("Menstrual History")
            col1, col2 = st.columns(2)
            with col1:
                form.menstrual_regular = st.selectbox("Regular?", ["", "Yes", "No"])
                form.menstrual_cycle = st.text_input("Cycle", form.menstrual_cycle)
            with col2:
                form.pms = st.text_input("PMS", form.pms)
                form.menstrual_flow = st.text_input("Flow", form.menstrual_flow)
    
    with tab5:
        form.medications_supplements = st.text_area("Medications/Supplements", 
                                                   form.medications_supplements, height=150)
        form.allergies = st.text_area("Allergies", form.allergies, height=100)
        form.past_medical_history = st.text_area("Medical History", 
                                               form.past_medical_history, height=100)
    
    return form

# ============================================
# MAIN APPLICATION
# ============================================

# Initialize transcriber
@st.cache_resource
def get_transcriber(api_key):
    if api_key:
        return MedicalTranscriber(api_key)
    return None

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: #0891B2; margin-bottom: 0;'>🏥 Clinical Scribe Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280; margin-top: 0;'>Dr. Shefali Verma - IRAC Dubai</p>", unsafe_allow_html=True)

# Settings in expandable section
with st.expander("⚙️ Settings & Patient Info", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        api_key = st.text_input("OpenAI API Key", type="password", help="Required for transcription")
        st.session_state.patient_name = st.text_input("Patient Name", value=st.session_state.patient_name)
    
    with col2:
        st.session_state.consultation_type = st.selectbox(
            "Consultation Type",
            ["General", "Functional Medicine Initial", "Functional Medicine Follow-up", 
             "Acute Visit", "Lab Review", "Telemedicine"]
        )
        save_audio = st.checkbox("Save audio recordings", value=False)

# Language selection
st.markdown("### 🌍 Patient Language")

languages = {
    "Auto-detect": "🌐",
    "English": "🇬🇧", 
    "Gulf Arabic": "🇦🇪",
    "Egyptian": "🇪🇬",
    "Levantine": "🇱🇧",
    "French": "🇫🇷",
    "Hindi": "🇮🇳"
}

cols = st.columns(len(languages))
for idx, (lang, flag) in enumerate(languages.items()):
    with cols[idx]:
        if st.button(f"{flag}\n{lang}", key=f"lang_{lang}", 
                    use_container_width=True,
                    type="primary" if st.session_state.selected_language == lang else "secondary"):
            st.session_state.selected_language = lang

# Main UI
if api_key:
    transcriber = get_transcriber(api_key)
    
    # Record button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if st.session_state.recording:
            if st.button("⏹️ STOP RECORDING", key="stop", use_container_width=True):
                st.session_state.recording = False
                if transcriber:
                    audio_data = transcriber.stop_recording()
        else:
            if st.button("🎤 START RECORDING", key="start", use_container_width=True):
                st.session_state.recording = True
                st.session_state.transcription = ""
                if transcriber:
                    transcriber.start_recording()
    
    # Recording indicator
    if st.session_state.recording:
        st.markdown('<div class="recording-indicator">● Recording in progress...</div>', unsafe_allow_html=True)
    
    # Live transcription display
    st.markdown("### 📝 Live Transcription")
    
    transcription_container = st.container()
    
    with transcription_container:
        if st.session_state.recording and transcriber:
            # Update with new transcribed text
            while not transcriber.text_queue.empty():
                new_text = transcriber.text_queue.get()
                st.session_state.transcription += " " + new_text
            
            # Display with medical term highlighting
            highlighted_text = transcriber.highlight_medical_terms(st.session_state.transcription)
            st.markdown(
                f'<div class="transcription-box">{highlighted_text if highlighted_text else "Listening..."}</div>',
                unsafe_allow_html=True
            )
            
            # Auto-detect dialect
            detected_dialect = transcriber.detect_dialect(st.session_state.transcription)
            if detected_dialect:
                dialect_names = {
                    'ar_gulf': 'Gulf Arabic detected',
                    'ar_egyptian': 'Egyptian Arabic detected',
                    'ar_levantine': 'Levantine Arabic detected',
                    'fr': 'French detected'
                }
                st.info(f"🔍 {dialect_names.get(detected_dialect, detected_dialect)}")
        else:
            # Editable transcription
            st.session_state.transcription = st.text_area(
                "Edit transcription:",
                value=st.session_state.transcription,
                height=200,
                label_visibility="collapsed"
            )
    
    # Functional Medicine Quick Add
    if "Functional Medicine" in st.session_state.consultation_type:
        with st.expander("🌿 Functional Medicine Quick Add"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.checkbox("Gut Health"):
                    gut_symptoms = st.multiselect(
                        "Symptoms:",
                        ["Bloating", "Gas", "Constipation", "Diarrhea", "Reflux", "Abdominal pain"]
                    )
            
            with col2:
                if st.checkbox("Hormonal"):
                    hormone_symptoms = st.multiselect(
                        "Symptoms:",
                        ["Fatigue", "Weight gain", "Hot flashes", "Low libido", "Mood swings", "Insomnia"]
                    )
            
            with col3:
                if st.checkbox("Protocols"):
                    protocols = st.multiselect(
                        "Discussed:",
                        ["Elimination Diet", "AIP", "Low FODMAP", "Detox", "Gut Healing", "Adrenal Support"]
                    )
    
    # IRAC Intake Form Integration
    if st.session_state.transcription:
        with st.expander("📋 IRAC Intake Form", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🤖 Auto-Fill from Transcript", use_container_width=True):
                    with st.spinner("Extracting information..."):
                        extractor = IntakeFormExtractor(api_key)
                        form = extractor.extract_form_data(st.session_state.transcription)
                        st.session_state.intake_form = form
                        st.session_state.show_intake_form = True
            
            with col2:
                if st.button("📝 Manual Entry", use_container_width=True):
                    st.session_state.intake_form = IRACIntakeForm()
                    st.session_state.show_intake_form = True
        
        # Show intake form if activated
        if st.session_state.show_intake_form and st.session_state.intake_form:
            # Identify missing information
            missing_info = []
            form = st.session_state.intake_form
            
            if not form.name:
                missing_info.append("Patient name")
            if not form.age and not form.date_of_birth:
                missing_info.append("Age/DOB")
            if not form.medications_supplements:
                missing_info.append("Current medications")
            if not form.allergies:
                missing_info.append("Allergies")
            
            # Render form
            form = render_intake_form_ui(form, missing_info)
            st.session_state.intake_form = form
            
            # Questions for missing info
            if missing_info:
                with st.expander("🎤 Questions to Ask"):
                    question_map = {
                        "Patient name": "Could you please tell me your full name?",
                        "Age/DOB": "What is your date of birth?",
                        "Current medications": "Are you taking any medications or supplements?",
                        "Allergies": "Do you have any allergies?"
                    }
                    for item in missing_info:
                        if item in question_map:
                            st.write(f"• {question_map[item]}")
            
            if st.button("💾 Save Intake Form", use_container_width=True):
                st.success("✅ Intake form saved!")
                st.session_state.show_intake_form = False
    
    # Generate Clinical Note
    if st.session_state.transcription and not st.session_state.recording:
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 GENERATE CLINICAL NOTE", key="generate", use_container_width=True):
                with st.spinner("Creating SOAP note..."):
                    client = openai.OpenAI(api_key=api_key)
                    
                    # Enhanced prompt for functional medicine
                    if "Functional" in st.session_state.consultation_type:
                        system_prompt = """You are Dr. Shefali Verma's medical scribe at IRAC Dubai, 
                        specializing in functional and integrative rheumatology. 
                        Create detailed SOAP notes that include root cause analysis, dietary recommendations, 
                        specific supplement protocols with dosages, and lifestyle modifications."""
                    else:
                        system_prompt = """You are Dr. Shefali Verma's medical scribe at IRAC Dubai.
                        Create professional clinical documentation."""
                    
                    # Include intake form data if available
                    intake_info = ""
                    if st.session_state.intake_form:
                        form = st.session_state.intake_form
                        intake_info = f"""
                        Patient Demographics:
                        - Name: {form.name}
                        - Age: {form.age}
                        - Gender: {form.gender}
                        
                        Vitals:
                        - BP: {form.blood_pressure}
                        - Pulse: {form.pulse_rate}
                        - Temperature: {form.temperature}
                        
                        Current Medications/Supplements: {form.medications_supplements}
                        Allergies: {form.allergies}
                        """
                    
                    prompt = f"""
                    Convert this medical consultation into a detailed SOAP note.
                    
                    Patient: {st.session_state.patient_name}
                    Consultation Type: {st.session_state.consultation_type}
                    Language: {st.session_state.selected_language}
                    
                    {intake_info}
                    
                    Transcript:
                    {st.session_state.transcription}
                    
                    Create a comprehensive SOAP note following Dr. Shefali Verma's style:
                    
                    SUBJECTIVE:
                    - Chief complaint with timeline
                    - History of present illness
                    - Review of systems
                    - Current medications and supplements
                    - Allergies
                    - Relevant lifestyle factors
                    
                    OBJECTIVE:
                    - Vital signs
                    - Physical exam findings
                    - Functional medicine observations
                    - Lab results if mentioned
                    
                    ASSESSMENT:
                    - Primary diagnosis
                    - Differential diagnoses
                    - Root cause analysis (for functional medicine)
                    - System imbalances identified
                    
                    PLAN:
                    - Medications with exact dosages
                    - Supplement protocol (brand names and dosages if mentioned)
                    - Dietary recommendations
                    - Lifestyle modifications
                    - Lab tests ordered
                    - Follow-up timeline
                    - Patient education provided
                    """
                    
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3
                        )
                        
                        st.session_state.clinical_note = response.choices[0].message.content
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Display Clinical Note
    if st.session_state.clinical_note:
        st.markdown("### 📋 Clinical Note")
        
        # Editable note
        edited_note = st.text_area(
            "Review and edit:",
            value=st.session_state.clinical_note,
            height=400,
            label_visibility="collapsed"
        )
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Save", use_container_width=True):
                # Save to database
                conn = sqlite3.connect('clinical_scribe.db')
                c = conn.cursor()
                
                intake_form_json = json.dumps(vars(st.session_state.intake_form)) if st.session_state.intake_form else None
                
                c.execute("""INSERT INTO consultations 
                           (timestamp, patient_name, language, dialect, consultation_type, 
                            transcription, clinical_note, intake_form, audio_path)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                         (datetime.now().isoformat(),
                          st.session_state.patient_name,
                          st.session_state.selected_language,
                          transcriber.detect_dialect(st.session_state.transcription) if transcriber else None,
                          st.session_state.consultation_type,
                          st.session_state.transcription,
                          edited_note,
                          intake_form_json,
                          None))
                
                conn.commit()
                conn.close()
                
                # Save to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"clinical_note_{st.session_state.patient_name.replace(' ', '_')}_{timestamp}.txt"
                
                os.makedirs("clinical_notes", exist_ok=True)
                with open(f"clinical_notes/{filename}", "w", encoding="utf-8") as f:
                    f.write(f"CLINICAL NOTE - DR. SHEFALI VERMA, IRAC DUBAI\n")
                    f.write(f"=" * 60 + "\n")
                    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Patient: {st.session_state.patient_name}\n")
                    f.write(f"Type: {st.session_state.consultation_type}\n")
                    f.write(f"Language: {st.session_state.selected_language}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(edited_note)
                    
                    if st.session_state.intake_form:
                        f.write("\n\n" + "=" * 60 + "\n")
                        f.write("INTAKE FORM DATA\n")
                        f.write("=" * 60 + "\n")
                        form = st.session_state.intake_form
                        f.write(f"Height: {form.height}, Weight: {form.weight}\n")
                        f.write(f"BP: {form.blood_pressure}, Pulse: {form.pulse_rate}\n")
                        f.write(f"Allergies: {form.allergies}\n")
                        f.write(f"Current Medications: {form.medications_supplements}\n")
                
                st.success(f"✅ Saved successfully!")
        
        with col2:
            if st.button("📋 Copy", use_container_width=True):
                st.code(edited_note, language=None)
                st.info("Select all and copy")
        
        with col3:
            if st.button("🔄 MedNet", use_container_width=True):
                st.session_state.show_mednet_export = True
        
        with col4:
            if st.button("🆕 New", use_container_width=True):
                st.session_state.transcription = ""
                st.session_state.clinical_note = ""
                st.session_state.patient_name = ""
                st.session_state.intake_form = None
                st.experimental_rerun()
        
        # MedNet Flow Export
        if st.session_state.get('show_mednet_export'):
            st.markdown("### 🔄 Export to MedNet Flow")
            
            integrator = MedNetFlowIntegration()
            
            # Prepare patient data
            patient_data = {
                'patient_name': st.session_state.patient_name,
                'consultation_type': st.session_state.consultation_type,
            }
            
            # Add intake form data if available
            if st.session_state.intake_form:
                form = st.session_state.intake_form
                patient_data.update({
                    'date_of_birth': form.date_of_birth,
                    'gender': form.gender,
                    'blood_group': form.blood_group,
                    'height': form.height,
                    'weight': form.weight,
                    'blood_pressure': form.blood_pressure,
                    'pulse_rate': form.pulse_rate,
                    'temperature': form.temperature,
                    'allergies': form.allergies,
                    'medications': form.medications_supplements
                })
            
            mednet_data = integrator.prepare_for_mednet(edited_note, patient_data, st.session_state.intake_form)
            
            col1, col2 = st.columns(2)
            
            with col1:
                export_format = st.selectbox(
                    "Export Format",
                    integrator.export_formats,
                    help="Choose how to transfer to MedNet"
                )
            
            with col2:
                if st.button("🚀 Generate Export", use_container_width=True):
                    st.session_state.mednet_export_ready = True
            
            if st.session_state.get('mednet_export_ready'):
                if export_format == "Clipboard":
                    clipboard_text = integrator.generate_clipboard_format(mednet_data)
                    
                    st.text_area(
                        "📋 Copy to MedNet:",
                        clipboard_text,
                        height=400
                    )
                    
                    if st.button("📋 Copy All", use_container_width=True):
                        try:
                            pyperclip.copy(clipboard_text)
                            st.success("✅ Copied! Paste in MedNet Flow")
                        except:
                            st.info("Select all text above and copy (Ctrl+C)")
                
                elif export_format == "Direct Entry Guide":
                    guide = integrator.generate_entry_guide(mednet_data)
                    
                    st.markdown("### 📖 MedNet Entry Guide")
                    st.code(guide, language=None)
                    
                    st.download_button(
                        "📄 Download Guide",
                        guide,
                        f"mednet_guide_{timestamp}.txt",
                        mime="text/plain"
                    )
            
            # MedNet tips
            with st.expander("💡 MedNet Flow Tips"):
                st.markdown("""
                **Quick Entry:**
                1. Open flow.mednetlabs.com in another tab
                2. Use Clipboard format for fastest entry
                3. Tab key moves between fields
                
                **Keyboard Shortcuts:**
                - F2: Save section
                - F3: Search medications
                - Tab: Next field
                """)

else:
    # Welcome screen
    st.info("👆 Please enter your OpenAI API key in Settings to begin")
    
    # Recent consultations
    st.markdown("### 📅 Recent Consultations")
    
    try:
        conn = sqlite3.connect('clinical_scribe.db')
        df = pd.read_sql_query("""
            SELECT timestamp, patient_name, consultation_type, language 
            FROM consultations 
            ORDER BY timestamp DESC 
            LIMIT 10
        """, conn)
        conn.close()
        
        if not df.empty:
            for _, row in df.iterrows():
                timestamp = datetime.fromisoformat(row['timestamp'])
                st.write(f"• **{row['patient_name']}** - {row['consultation_type']} ({row['language']}) - {timestamp.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.write("No recent consultations")
    except Exception as e:
        st.write("No consultations yet")
    
    # Quick start guide
    with st.expander("📖 Quick Start Guide"):
        st.markdown("""
        ### How to Use Clinical Scribe Pro
        
        1. **Enter API Key**: Get from OpenAI (one-time setup)
        2. **Patient Info**: Enter name and consultation type
        3. **Select Language**: Tap the flag for patient's language
        4. **Start Recording**: Big red button to begin
        5. **Watch Transcription**: See words appear live
        6. **Stop & Generate**: Create professional SOAP note
        7. **Review & Save**: Edit if needed, then save
        8. **Export to MedNet**: Copy-paste into flow.mednetlabs.com
        
        ### Features:
        - 🌍 **Multi-language**: Arabic dialects, French, Hindi, English
        - 🌿 **Functional Medicine**: Specialized for Dr. Shefali's practice
        - 📋 **IRAC Forms**: Auto-fill intake forms
        - 🔄 **MedNet Integration**: Easy export to EMR
        - 💾 **Auto-save**: Never lose your work
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #9CA3AF; font-size: 14px;'>Clinical Scribe Pro v2.0 • IRAC Dubai • Built with ❤️ for Dr. Shefali Verma</p>",
    unsafe_allow_html=True
)
