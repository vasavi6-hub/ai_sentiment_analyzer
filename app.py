import streamlit as st
from transformers import pipeline

# Title and Description
st.title("AI Sentiment Analysis Tool 🤖")
st.write("Enter any text below to see if the AI thinks it's Positive or Negative.")

# Load the AI model (this happens in the cloud)
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

# User Input
user_input = st.text_area("Type your text here:", "I am really excited about this AI project!")

if st.button("Analyze Sentiment"):
    with st.spinner('AI is thinking...'):
        result = classifier(user_input)[0]
        label = result['label']
        score = result['score']
        
        # Display results with colors
        if label == "POSITIVE":
            st.success(f"Result: {label} (Confidence: {score:.2f})")
        else:
            st.error(f"Result: {label} (Confidence: {score:.2f})")
