
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from tqdm import tqdm
import numpy as np
import logging

# Suppress HF warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

def analyze_bias(data_dir, model_path):
    print(f"Loading data from {data_dir} and model from {model_path}...")
    
    # Load Test Data
    test_path = os.path.join(data_dir, "processed/test.csv")
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found.")
        return
    test_df = pd.read_csv(test_path)
    
    # Load Metadata for rich features
    meta_path = os.path.join(data_dir, "raw/metadata.csv")
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found.")
        return
    meta_df = pd.read_csv(meta_path)
    
    # Merge
    # Ensure item_id is string/int consistent
    test_df['item_id'] = test_df['item_id'].astype(str)
    meta_df['itemid'] = meta_df['itemid'].astype(str)
    
    # We only care about metadata for the test set
    merged_df = test_df.merge(meta_df, left_on='item_id', right_on='itemid', how='left')
    
    print(f"Loaded {len(merged_df)} test cases. Merged with metadata.")
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.to(device)
    model.eval()
    
    # Predict
    predictions = []
    
    print("Running inference on test set...")
    # Batch processing would be faster but for ~80 items loop is fine and safer for memory on CPU
    for text in tqdm(merged_df['text']):
        inputs = tokenizer(
            str(text), 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred)
            
    merged_df['prediction'] = predictions
    merged_df['correct'] = (merged_df['prediction'] == merged_df['label'])
    
    # --- ANALYSIS ---
    
    # 1. Respondent Bias
    print("\n" + "="*30)
    print("RESPONDENT BIAS ANALYSIS")
    print("="*30)
    if 'respondent' in merged_df.columns:
        # Split multiple respondents (e.g. "RUS;UKR")? For now take whole string
        respondent_stats = merged_df.groupby('respondent').agg(
            count=('item_id', 'count'),
            accuracy=('correct', 'mean'),
            pred_violation_rate=('prediction', 'mean'),
            actual_violation_rate=('label', 'mean')
        ).sort_values('count', ascending=False)
        
        print("\nStats by Respondent State (Top 10 by count):")
        print(respondent_stats.head(10).to_string())
        
        # Check for extreme deviation
        # e.g. Country has 0% or 100% violation rate predicted vs actual
    else:
        print("No 'respondent' column found in metadata.")

    # 2. Temporal Bias
    print("\n" + "="*30)
    print("TEMPORAL BIAS ANALYSIS")
    print("="*30)
    if 'judgementdate' in merged_df.columns:
        merged_df['year'] = pd.to_datetime(merged_df['judgementdate'], errors='coerce').dt.year
        year_stats = merged_df.groupby('year').agg(
            count=('item_id', 'count'),
            accuracy=('correct', 'mean'),
            pred_violation_rate=('prediction', 'mean'),
            actual_violation_rate=('label', 'mean')
        ).sort_values('year')
        
        print("\nStats by Judgment Year:")
        print(year_stats.to_string())
        
        # Simple correlation
        valid_years = merged_df.dropna(subset=['year'])
        if len(valid_years['prediction'].unique()) > 1:
            corr = valid_years['year'].corr(valid_years['prediction'])
            print(f"\nCorrelation (Year vs Predicted Violation): {corr:.4f}")
        else:
            print("\nCorrelation (Year vs Predicted Violation): N/A (No prediction variance)")
    else:
        print("No 'judgementdate' column found.")

    # 3. Length Bias
    print("\n" + "="*30)
    print("TEXT LENGTH BIAS ANALYSIS")
    print("="*30)
    merged_df['text_len'] = merged_df['text'].apply(lambda x: len(str(x)))
    
    # Correlation
    if len(merged_df['prediction'].unique()) > 1:
        corr_pred = merged_df['text_len'].corr(merged_df['prediction'])
        print(f"Correlation (Text Length vs Predicted Violation): {corr_pred:.4f}")
    else:
         print("Correlation (Text Length vs Predicted Violation): N/A (No prediction variance)")
         
    if len(merged_df['correct'].unique()) > 1:
        corr_acc = merged_df['text_len'].corr(merged_df['correct'])
        print(f"Correlation (Text Length vs Accuracy): {corr_acc:.4f}")
    else:
        print("Correlation (Text Length vs Accuracy): N/A (Constant Accuracy)")
    
    len_v = merged_df[merged_df['prediction']==1]['text_len'].mean()
    len_nv = merged_df[merged_df['prediction']==0]['text_len'].mean()
    print(f"\nAvg Text Length (Predicted Violation): {len_v:.1f} chars")
    if pd.notna(len_nv):
        print(f"Avg Text Length (Predicted Non-Violation): {len_nv:.1f} chars")
    else:
        print("Avg Text Length (Predicted Non-Violation): N/A (None predicted)")

if __name__ == "__main__":
    analyze_bias("data", "results/final_model")
