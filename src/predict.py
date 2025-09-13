# src/predict.py
import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from bertopic import BERTopic
import joblib

print("Loading all model components... This may take a moment.")

# --- 1. Load All Saved Model Components ---
device = 0 if torch.cuda.is_available() else -1
emotion_classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
    device=device
)
topic_model = BERTopic.load("saved_models/bertopic_model")
tokenizer = AutoTokenizer.from_pretrained("saved_models/deberta_semantic_core")
deberta_model = AutoModel.from_pretrained("saved_models/deberta_semantic_core").to(device)
lgbm_model = joblib.load("saved_models/lgbm_meta_learner.pkl")


# --- 2. Load Feature Information from Training (CORRECTED) ---

print("Loading feature information...")
# --- FIX IS HERE ---
# Load the full topic column to learn about ALL possible topics from the training data.
topic_schema_df = pd.read_csv('data/enriched_data.csv', usecols=['topic'])
topic_cols = pd.get_dummies(topic_schema_df['topic'], prefix='topic').columns.tolist()

# Load just the first row to get the emotion column names efficiently.
df_schema = pd.read_csv('data/enriched_data.csv', nrows=1)
emotion_cols = [col for col in df_schema.columns if col not in ['sentiment', 'text', 'tweetid', 'label', 'topic']]


# --- 3. Create the Prediction Function ---

def get_deberta_embeddings(text, model, tokenizer):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(model.device)
        outputs = model(**inputs, output_hidden_states=True)
        cls_embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
    return cls_embedding

def predict_sentiment(text: str):
    text_list = [text]
    semantic_features = get_deberta_embeddings(text, deberta_model, tokenizer)
    emotion_results = emotion_classifier(text_list, truncation=True)
    emotion_scores = {item['label']: item['score'] for item in emotion_results[0]}
    emotion_features = np.array([emotion_scores.get(col, 0) for col in emotion_cols]).reshape(1, -1)
    
    predicted_topic, _ = topic_model.transform(text_list)
    topic_df = pd.DataFrame({'topic': predicted_topic})
    # The reindex command will now work correctly because `topic_cols` contains all possible topics.
    topic_one_hot = pd.get_dummies(topic_df['topic'], prefix='topic').reindex(columns=topic_cols, fill_value=0)
    topic_features = topic_one_hot.values

    final_feature_vector = np.concatenate([semantic_features, emotion_features, topic_features], axis=1)

    prediction = lgbm_model.predict(final_feature_vector)
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    return sentiment_labels[prediction[0]]


# --- 4. Use the Model ---

if __name__ == "__main__":
    print("\n✅ Model loaded. Ready for prediction.")
    
    while True:
        user_message = input("\nEnter a message to analyze (or type 'quit' to exit): ")
        if user_message.lower() == 'quit':
            break
        predicted_sentiment = predict_sentiment(user_message)
        print(f"--> Predicted Sentiment: {predicted_sentiment}")