import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="28-Emotion AI Analyzer", layout="wide")

st.title("Deep Emotion Intelligence 🧠")
st.write("This AI detects 28 different emotions from the GoEmotions dataset.")

# Load the 28-emotion model
@st.cache_resource
def load_emotion_model():
    # 'top_k=None' tells the AI to return EVERY emotion, not just one
    return pipeline("sentiment-analysis", model="arpanghoshal/EmoRoBERTa", top_k=None)

classifier = load_emotion_model()

user_input = st.text_input("Enter a sentence to analyze:", "I am so grateful for your help, it makes me very happy!")

if user_input:
    results = classifier(user_input)[0]
    
    # Show the top emotion in a big highlight box
    top_emotion = results[0]['label'].upper()
    st.subheader(f"Primary Emotion: {top_emotion}")
    
    # Create columns to show all emotions with progress bars
    st.write("### All Detected Emotions:")
    col1, col2 = st.columns(2)
    
    # Split the 28 emotions into two columns for better visibility
    for i, res in enumerate(results):
        label = res['label']
        score = res['score']
        
        target_col = col1 if i < 14 else col2
        with target_col:
            st.write(f"**{label.title()}** ({score:.1%})")
            st.progress(score)

