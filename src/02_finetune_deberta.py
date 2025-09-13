# src/02_finetune_deberta.py
import pandas as pd
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import numpy as np
import torch

def finetune_deberta():
    """
    Fine-tunes the DeBERTa model on the text data for semantic understanding.
    """
    print("--- Phase 2: DeBERTa Fine-Tuning Starting (using 'base' model) ---")
    print("NOTE: This will be much faster, estimated to take 1.5 - 3 hours.")
    
    print("Loading enriched data...")
    df = pd.read_csv('data/enriched_data.csv')

    sentiment_mapping = {-1: 0, 0: 1, 1: 2}
    df['label'] = df['sentiment'].map(sentiment_mapping)

    dataset = Dataset.from_pandas(df[['text', 'label']])
    dataset = dataset.cast_column("label", ClassLabel(num_classes=3, names=['Negative', 'Neutral', 'Positive']))

    train_test_split = dataset.train_test_split(test_size=0.2, stratify_by_column='label')
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    print("Tokenizing data...")
    # --- CHANGE 1: Switched to the smaller 'base' model ---
    MODEL_NAME = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    print("Loading DeBERTa model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        return {"accuracy": accuracy["accuracy"], "f1_weighted": f1["f1"]}

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        learning_rate=2e-5,
        # --- CHANGE 2: Increased batch size, as 'base' model uses less memory ---
        per_device_train_batch_size=16,
        # No longer need gradient_accumulation_steps or adafactor with the smaller model
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting fine-tuning...")
    trainer.train()

    print("Saving the best model to saved_models/deberta_semantic_core...")
    trainer.save_model("saved_models/deberta_semantic_core")
    tokenizer.save_pretrained("saved_models/deberta_semantic_core")
    print("--- Phase 2 Complete ---")

if __name__ == "__main__":
    finetune_deberta()