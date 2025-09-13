# src/03_train_meta_learner.py
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import joblib

def get_deberta_embeddings(texts, model, tokenizer):
    """
    Extracts the [CLS] token embedding from the last hidden state of DeBERTa.
    These embeddings are the powerful semantic features.
    """
    print("Extracting DeBERTa embeddings for the meta-learner...")
    model.eval() # Set the model to evaluation mode
    all_embeddings = []
    device = model.device
    
    with torch.no_grad(): # Disable gradient calculation for faster inference
        # Process in batches to avoid using too much memory
        for i in range(0, len(texts), 64): 
            batch = texts[i:i+64]
            # Tokenize the batch and send it to the GPU
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            # Get model outputs, ensuring we get the hidden states
            outputs = model(**inputs, output_hidden_states=True)
            # We take the embedding of the [CLS] token from the last hidden layer as the sentence representation
            cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
            
    return np.vstack(all_embeddings)

def train_meta_learner():
    """
    Combines all features and trains the final LightGBM model using robust cross-validation.
    """
    print("--- Phase 3: Meta-Learner Training Starting ---")
    
    # --- 1. Load Data and Models ---
    print("Loading enriched data and fine-tuned models...")
    df = pd.read_csv('data/enriched_data.csv')
    
    sentiment_mapping = {-1: 0, 0: 1, 1: 2}
    df['label'] = df['sentiment'].map(sentiment_mapping)

    # Load the base DeBERTa model (not the classification head) for embedding extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("saved_models/deberta_semantic_core")
    deberta_model = AutoModel.from_pretrained("saved_models/deberta_semantic_core").to(device)
    
    # --- 2. Create the Final Feature Matrix ---
    print("Assembling final feature matrix...")
    # a. Semantic features from DeBERTa
    semantic_features = get_deberta_embeddings(df['text'].tolist(), deberta_model, tokenizer)
    
    # b. Emotion features from Phase 1 (dynamically select columns)
    emotion_cols = [col for col in df.columns if col not in ['sentiment', 'text', 'tweetid', 'label', 'topic']]
    emotion_features = df[emotion_cols].values
    
    # c. Topic features from Phase 1 (one-hot encoded for the tree model)
    topic_features = pd.get_dummies(df['topic'], prefix='topic').values
    
    # Combine all features horizontally into a single NumPy array
    X = np.concatenate([semantic_features, emotion_features, topic_features], axis=1)
    y = df['label'].values
    print(f"Final feature matrix shape: {X.shape}")

    # --- 3. Train with Stratified K-Fold Cross-Validation ---
    print("Training meta-learner with 5-Fold Cross-Validation...")
    # Cross-validation gives a much more reliable estimate of the model's true performance
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    reports = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1}/5 ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        lgbm = lgb.LGBMClassifier(objective='multiclass', random_state=42, n_estimators=500)
        lgbm.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 eval_metric='multi_logloss',
                 callbacks=[lgb.early_stopping(10, verbose=False)]) # Stops training if performance doesn't improve
        
        preds = lgbm.predict(X_val)
        report = classification_report(y_val, preds, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)
        reports.append(report)
        models.append(lgbm)
        print(f"Fold {fold+1} Accuracy: {report['accuracy']:.4f}")
    
    # --- 4. Final Evaluation and Saving ---
    print("\n--- Final Model Performance (Averaged over 5 Folds) ---")
    avg_accuracy = np.mean([r['accuracy'] for r in reports])
    avg_f1 = np.mean([r['weighted avg']['f1-score'] for r in reports])
    print(f"Average Cross-Validation Accuracy: {avg_accuracy:.4f}")
    print(f"Average Cross-Validation Weighted F1-Score: {avg_f1:.4f}")
    
    # Save one of the trained models for future use
    joblib.dump(models[0], "saved_models/lgbm_meta_learner.pkl")
    print("\nSaved one of the trained LightGBM models to saved_models/lgbm_meta_learner.pkl")
    print("--- Phase 3 Complete ---")

if __name__ == "__main__":
    train_meta_learner()