import streamlit as st
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import zipfile
import io
import os
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm

# Set page config
st.set_page_config(
    page_title="Advanced AI Text Detector",
    page_icon="🔍",
    layout="wide"
)

# Constants
MODEL_NAME = "roberta-base"
MAX_LENGTH = 512
BATCH_SIZE = 8
DATASETS_DIR = "./datasets"
MODEL_PATH = "./ai_detector_model.pt"

# Define a PyTorch Dataset
class TextDetectionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Download and prepare datasets
@st.cache_resource
def download_datasets():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    # Load datasets from Hugging Face
    try:
        # Human text from Wikipedia
        wiki_data = load_dataset("wikipedia", "20220301.en", split="train[:5000]")
        human_texts = [entry['text'] for entry in wiki_data if len(entry['text'].split()) > 30][:1000]
        
        # Human text from BookCorpus
        books_data = load_dataset("bookcorpus", split="train[:5000]")
        human_texts.extend([entry['text'] for entry in books_data if len(entry['text'].split()) > 30][:1000])
        
        # AI generated text
        ai_data = load_dataset("openai/webgpt_comparisons", split="train[:2000]")
        ai_texts = [entry['answer'] for entry in ai_data if len(entry['answer'].split()) > 30][:2000]
        
        # Labels: 1 for AI, 0 for human
        human_labels = [0] * len(human_texts)
        ai_labels = [1] * len(ai_texts)
        
        all_texts = human_texts + ai_texts
        all_labels = human_labels + ai_labels
        
        # Save as CSV
        data_df = pd.DataFrame({'text': all_texts, 'label': all_labels})
        data_df.to_csv(os.path.join(DATASETS_DIR, 'text_detection_dataset.csv'), index=False)
        
        return all_texts, all_labels
    
    except Exception as e:
        st.error(f"Error downloading datasets: {str(e)}")
        # Fallback to generate synthetic data if datasets fail to download
        return generate_synthetic_data()

# Fallback function to generate synthetic data if datasets can't be loaded
def generate_synthetic_data():
    # Human-like texts
    human_texts = [
        "I went to the store yesterday to buy some groceries. The weather was terrible, it was raining cats and dogs.",
        "My favorite book series has to be Harry Potter. I've read all seven books at least three times each.",
        "I'm not sure what to do about my car. It's making a strange noise whenever I turn left.",
        "The concert last night was amazing! The band played for nearly three hours and the crowd was so energetic.",
        "I'm thinking about taking a vacation next month. I'm torn between going to the beach or the mountains.",
        "My grandmother's recipe for apple pie is the best. The secret is adding a bit of lemon zest to the filling.",
        "I overslept this morning and was late for my meeting. My boss wasn't happy about it.",
        "I can't believe how much my nephew has grown since I last saw him. Kids grow up so fast!",
        "The restaurant we tried last weekend was a big disappointment. The service was slow and the food was mediocre.",
        "I've been trying to learn Spanish for the past six months, but I'm still struggling with the subjunctive tense."
    ] * 10  # Repeat to get more samples
    
    # AI-like texts
    ai_texts = [
        "The integration of artificial intelligence in healthcare systems presents both opportunities and challenges for medical practitioners and patients alike.",
        "Recent advancements in quantum computing have enabled researchers to solve previously intractable problems in cryptography and materials science.",
        "The correlation between socioeconomic factors and educational outcomes has been extensively studied, revealing complex patterns of causation and association.",
        "An analysis of climate data from the past century demonstrates a clear trend of increasing global temperatures and corresponding changes in weather patterns.",
        "The implementation of machine learning algorithms in financial services has revolutionized risk assessment and fraud detection methodologies.",
        "A comprehensive review of the literature suggests that multimodal approaches to neurological rehabilitation yield superior outcomes compared to unimodal interventions.",
        "The philosophical implications of consciousness in artificial systems raise profound questions about the nature of sentience and ethical considerations thereof.",
        "Economic modeling indicates that targeted investment in renewable energy infrastructure could simultaneously address climate concerns and stimulate job growth.",
        "The biomechanical analysis of human locomotion provides insights into the optimization of prosthetic designs and rehabilitation protocols.",
        "A systematic evaluation of educational technologies reveals varying degrees of efficacy across different demographic groups and learning contexts."
    ] * 10  # Repeat to get more samples
    
    human_labels = [0] * len(human_texts)
    ai_labels = [1] * len(ai_texts)
    
    all_texts = human_texts + ai_texts
    all_labels = human_labels + ai_labels
    
    # Save as CSV
    data_df = pd.DataFrame({'text': all_texts, 'label': all_labels})
    data_df.to_csv(os.path.join(DATASETS_DIR, 'synthetic_dataset.csv'), index=False)
    
    return all_texts, all_labels

# Load or train the model
@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Check if we have a trained model already
    if os.path.exists(MODEL_PATH):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model, tokenizer
    
    # If no model exists, train one
    texts, labels = download_datasets()
    
    # Split the data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    train_dataset = TextDetectionDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TextDetectionDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Training loop
    num_epochs = 3
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = correct / total
        
    # Save the model
    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
    
    return model, tokenizer

# Make prediction using transformer model
def predict_with_transformer(text, model, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Tokenize the input
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    # Get prediction and confidence
    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prediction].item() * 100
    
    return prediction, confidence, probabilities[0].tolist()

# Function to extract linguistic features for explanation
def extract_linguistic_features(text):
    features = {}
    
    # Word count
    words = text.split()
    features['word_count'] = len(words)
    
    # Sentence count
    sentences = [s for s in text.split('.') if s.strip()]
    features['sentence_count'] = len(sentences)
    
    # Average words per sentence
    if sentences:
        features['avg_words_per_sentence'] = features['word_count'] / features['sentence_count']
    else:
        features['avg_words_per_sentence'] = 0
    
    # Unique words ratio
    if words:
        features['unique_words_ratio'] = len(set(words)) / len(words)
    else:
        features['unique_words_ratio'] = 0
    
    # Average word length
    if words:
        features['avg_word_length'] = sum(len(word) for word in words) / len(words)
    else:
        features['avg_word_length'] = 0
    
    return features

# Main Streamlit app
def main():
    st.title("🔍 Advanced AI Text Detector")
    st.markdown("""
    ### Detect if text was written by AI or a human
    
    This app uses a RoBERTa-based transformer model trained on datasets of human-written and 
    AI-generated text to determine whether input text was likely written by a human or generated by AI.
    """)
    
    # Display loading spinner while preparing model
    with st.spinner("Loading transformer model... This may take a moment."):
        try:
            model, tokenizer = get_model()
            model_loaded = True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            model_loaded = False
    
    if model_loaded:
        st.success("Model loaded successfully!")
        
        # Text input
        text_input = st.text_area("Enter text to analyze:", height=200)
        
        # Analysis button
        if st.button("Analyze Text"):
            if not text_input or len(text_input.strip()) < 20:
                st.warning("Please enter at least 20 characters of text to analyze.")
            else:
                with st.spinner("Analyzing text with transformer model..."):
                    # Make prediction
                    prediction, confidence, probabilities = predict_with_transformer(text_input, model, tokenizer)
                    
                    # Get additional analysis
                    linguistic_features = extract_linguistic_features(text_input)
                    
                    # Display results
                    st.markdown("## Analysis Results")
                    
                    # Create columns for layout
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Display prediction result
                        if prediction == 1:
                            source = "AI-Generated"
                            emoji = "🤖"
                            color = "#FF9494"  # Light red
                        else:
                            source = "Human-Written"
                            emoji = "👤"
                            color = "#94FF94"  # Light green
                        
                        st.markdown(f"### {emoji} Text appears to be: {source}")
                        
                        # Create confidence meter
                        st.progress(confidence/100)
                        st.markdown(f"Confidence: {confidence:.1f}%")
                        
                        # Probability distribution
                        prob_df = pd.DataFrame({
                            'Source': ['Human', 'AI'],
                            'Probability': [probabilities[0], probabilities[1]]
                        })
                        
                        st.markdown("### Probability Distribution")
                        st.bar_chart(prob_df.set_index('Source'))
                        
                        # Analysis explanation
                        st.markdown("### Analysis Explanation")
                        if prediction == 1:
                            st.markdown("""
                            **Why this might be AI-generated:**
                            - Consistent structure and patterned language
                            - High lexical diversity and formal vocabulary
                            - Even distribution of sentence lengths and complexity
                            - Logical flow with fewer natural digressions
                            - Less varied punctuation and speech patterns
                            """)
                        else:
                            st.markdown("""
                            **Why this might be human-written:**
                            - Natural variations in language patterns
                            - More personal and subjective language
                            - Varied sentence structures and informal elements
                            - Unique stylistic choices and idiosyncrasies
                            - More varied punctuation and speech patterns
                            """)
                    
                    with col2:
                        # Display text statistics
                        st.markdown("### Text Statistics")
                        stats_df = pd.DataFrame({
                            'Metric': [
                                'Word Count',
                                'Sentence Count',
                                'Avg Words/Sentence',
                                'Unique Words Ratio',
                                'Avg Word Length'
                            ],
                            'Value': [
                                linguistic_features['word_count'],
                                linguistic_features['sentence_count'],
                                f"{linguistic_features['avg_words_per_sentence']:.1f}",
                                f"{linguistic_features['unique_words_ratio']:.2f}",
                                f"{linguistic_features['avg_word_length']:.1f}"
                            ]
                        })
                        st.dataframe(stats_df, hide_index=True)
                        
                        # Tips based on result
                        st.markdown("### Tips")
                        if prediction == 1:
                            st.markdown("""
                            If you want your AI-generated text to appear more human:
                            - Add more personal experiences and opinions
                            - Vary your sentence structures more
                            - Include some writing quirks or informalities
                            - Add conversational elements and transitions
                            """)
                        else:
                            st.markdown("""
                            Characteristics that helped identify this as human text:
                            - Personal voice and perspective
                            - Natural flow with varied structures
                            - Authentic expression patterns
                            - Stylistic choices typical of human writing
                            """)
    
    # Add information section
    st.markdown("---")
    st.markdown("""
    ### How it Works
    
    This model uses RoBERTa, a transformer-based language model, trained on:
    - Human-written text from Wikipedia and BookCorpus
    - AI-generated text from various sources
    
    The model analyzes patterns in language use, sentence structure, vocabulary diversity, 
    and other linguistic features that differ between human and AI-generated content.
    
    **Note:** While this model is more advanced than simple rule-based detectors, no AI detection 
    system is 100% accurate. Very high-quality AI text may be classified as human-written, 
    and some human writing may be incorrectly flagged as AI-generated.
    """)

if __name__ == "__main__":
    main()
