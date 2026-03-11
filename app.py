import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import time
import pandas as pd
import json
import os
from google import genai
from google.cloud import storage
import logging
import warnings

# Suppress HuggingFace and local warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="QuantumLeap Insights AI",
    layout="wide"
)

# ==========================================
# Caching Models to improve performance
# ==========================================
class MultiTaskBert(nn.Module):
    def __init__(self, num_sentiments=3, num_categories=5):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.sent_head = nn.Linear(self.bert.config.hidden_size, num_sentiments)
        self.cat_head = nn.Linear(self.bert.config.hidden_size, num_categories)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        sentiment_logits = self.sent_head(pooled_output)
        category_logits = self.cat_head(pooled_output)
        return sentiment_logits, category_logits

@st.cache_resource
def load_models():
    """Loads and caches the heavy transformer models to prevent reloading on every run."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer and initialize model
    multitask_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    multitask_model = MultiTaskBert().to(device)

    MODEL_FILE = "multitask_bert.pth"
    BUCKET_NAME = "run-sources-project-bd9685af-3983-407e-a25-us-central1"
    BLOB_PATH = "services/streamlit-app/multitask_bert.pth"

    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILE)

    # Download model if it does not exist locally
    if not os.path.exists(model_path):
        try:
            st.info("Downloading model from Cloud Storage...")

            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(BLOB_PATH)
            blob.download_to_filename(model_path)

            st.success("Model downloaded successfully.")

        except Exception as e:
            st.warning(f"Could not download multitask_bert.pth: {e}")

    # Load weights if available
    if os.path.exists(model_path):
        multitask_model.load_state_dict(torch.load(model_path, map_location=device))
        multitask_model.eval()
    else:
        st.warning("Model file still not found. Using untrained Bert Model.")

    return device, multitask_tokenizer, multitask_model

# Predict function to replace mock_classification
def predict_classification(text, tokenizer, model, device):
    categories = [
        "Bug Report",
        "Feature Request",
        "Usability Issue",
        "Performance Complaint",
        "Positive Feedback"
    ]

    model.eval()
    encoded = tokenizer(text, return_tensors='pt', max_length=64, truncation=True, padding='max_length')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        sent_logits, cat_logits = model(input_ids, attention_mask)

        # Determine category
        pred_cat_idx = torch.argmax(cat_logits, dim=1).item()
        pred_cat = categories[pred_cat_idx]

        # Determine sentiment
        sent_val = torch.argmax(sent_logits, dim=1).item()
        pred_sent = "Positive" if sent_val == 2 else ("Neutral" if sent_val == 1 else "Negative")

    return {
        "Category": pred_cat,
        "Sentiment": pred_sent
    }

def extract_entities_gemini(review_text, api_key):
    if not api_key:
        return {
            "Affected Feature": "N/A (Missing API Key)",
            "Severity": "N/A",
            "Raw_Output": "Please provide a Gemini API Key."
        }
    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""
        Extract the affected product feature and the severity of the issue from the following customer feedback.
        Customer Feedback: "{review_text}"
        Return ONLY a valid JSON object with exactly two keys: "Affected Feature" and "Severity".
        Example: {{"Affected Feature": "PDF Export", "Severity": "High"}}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )

        # Clean up the output to parse JSON: removing code blocks and newlines if they exist
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        cleaned_text = raw_text.strip()

        parsed = json.loads(cleaned_text)

        return {
            "Affected Feature": parsed.get("Affected Feature", "Unknown"),
            "Severity": parsed.get("Severity", "Unknown"),
            "Raw_Output": cleaned_text
        }
    except Exception as e:
        return {
            "Affected Feature": "Error",
            "Severity": "Error",
            "Raw_Output": f"Gemini API Error: {str(e)}"
        }

def generate_innovative_features_gemini(complaint_summary, api_key):
    if not api_key:
        return "Note: Please provide a Gemini API Key in the sidebar to generate feature proposals."
    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""Based on the following customer feedback, suggest exactly 3 clear and actionable solutions to resolve the user's issue.

        Customer Feedback: "{complaint_summary}"

        IMPORTANT: Keep your response extremely concise. Output exactly three brief bullet points (maximum 1-2 sentences per point). Do not include any introductory or concluding text.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error resolving Gemini API response: {str(e)}"

# Sidebar for API Configuration
st.sidebar.header("Configuration")

# Securely load from Environment Variable if deployed on Google Cloud
env_api_key = os.environ.get("API_KEY", "")

if env_api_key:
    st.sidebar.success("Gemini API Key securely loaded from Cloud Environment.")
    gemini_key = env_api_key
else:
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="Providing your Google Gemini API key enables generative feature proposals.")

# Load Models
with st.spinner("Loading AI Models (this may take a minute)..."):
    device, multitask_tokenizer, multitask_model = load_models()

# ==========================================
# UI Layout
# ==========================================
st.title("QuantumLeap Innovations")
st.subheader("AI-Powered Customer Insight & Feature Generation")

st.markdown("""
This application analyzes raw customer feedback using a multi-task NLP architecture and automatically generates actionable product feature concepts using the **Gemini API**.
""")

st.divider()

# Input Section
st.header("1. Input Customer Feedback")
sample_text = "The analytics dashboard crashes every time I try exporting reports in PDF format. It takes way too long."
user_input = st.text_area("Paste customer review or support ticket here:", value=sample_text, height=150)

if st.button("Analyze Pipeline", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        st.divider()
        st.header("2. AI Analysis Insights")

        # Progress simulation
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Classifying feedback...")
        time.sleep(0.5)
        progress_bar.progress(33)
        class_res = predict_classification(user_input, multitask_tokenizer, multitask_model, device)

        status_text.text("Extracting key entities (Gemini)...")
        time.sleep(0.5)
        # Using the gemini key declared later (or from sidebar)
        entities = extract_entities_gemini(user_input, gemini_key)
        progress_bar.progress(66)

        status_text.text("Generating feature proposals (Gemini)...")
        features = generate_innovative_features_gemini(user_input, gemini_key)
        progress_bar.progress(100)
        status_text.empty()

        # Output Section - Columns layout
        col1, col2 = st.columns(2)

        with col1:
            st.info("### Core Classification")
            st.metric(label="Predicted Category", value=class_res["Category"])
            st.metric(label="Sentiment", value=class_res["Sentiment"])

        with col2:
            st.warning("### Extracted Entities (Gemini API)")
            st.write(f"**Affected Target:** {entities['Affected Feature']}")
            st.write(f"**Urgency/Severity:** {entities['Severity']}")
            st.write(f"*Raw Output:* {entities.get('Raw_Output', '')}")

        st.success("### Generative Feature Proposals (Gemini API)")
        st.markdown(features)

        # View raw JSON option
        with st.expander("View Raw JSON Output"):
            final_output = {
                "User Feedback": user_input,
                "Classified Category": class_res["Category"],
                "Sentiment Level": class_res["Sentiment"],
                "Extracted Entities": {
                    "Affected Target": entities['Affected Feature'],
                    "Severity": entities['Severity']
                },
                "LLM Product Suggestions": features
            }
            st.json(final_output)

st.divider()
st.markdown("*Prototype built for QuantumLeap Innovations using Streamlit and HuggingFace Transformers.*")
