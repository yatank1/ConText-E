# src/01_feature_engineering.py
import pandas as pd
from transformers import pipeline
from bertopic import BERTopic
import torch

def generate_features(input_path, output_path):
    print("--- Phase 1: Feature Engineering Starting ---")
    print("Loading raw data...")
    df = pd.read_csv(input_path)

    # Basic Cleaning
    df = df[df['sentiment'].isin([-1, 0, 1])].copy()
    df.rename(columns={'message': 'text'}, inplace=True)
    df.dropna(subset=['text'], inplace=True)
    docs = df['text'].tolist()

    # --- Generate Emotional Features ---
    print("Generating emotional features... (This may take 15-30 minutes)")
    device = 0 if torch.cuda.is_available() else -1
    emotion_classifier = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,
        device=device
    )
    # The fix is adding `truncation=True` to handle tweets longer than the model's limit.
    emotion_results = emotion_classifier(docs, batch_size=128, truncation=True) # <-- FIX IS HERE
    
    print("Processing emotion results...")
    emotion_df = pd.json_normalize([
        {item['label']: item['score'] for item in result} 
        for result in emotion_results
    ])
    emotion_df.fillna(0, inplace=True)
    
    # --- Generate Topical Features ---
    print("Generating topical features... (This may take 20-40 minutes)")
    topic_model = BERTopic(verbose=True, min_topic_size=50) 
    topics, _ = topic_model.fit_transform(docs)
    
    # --- Combine and Save ---
    print("Combining all features and saving...")
    df['topic'] = topics
    enriched_df = pd.concat([df.reset_index(drop=True), emotion_df], axis=1)
    
    enriched_df.to_csv(output_path, index=False)
    print(f"Enriched data saved to {output_path}")
    
    topic_model.save("saved_models/bertopic_model", serialization="safetensors")
    print("--- Phase 1 Complete ---")

if __name__ == "__main__":
    generate_features(
        input_path='data/twitter_sentiment_data.csv',
        output_path='data/enriched_data.csv'
    )