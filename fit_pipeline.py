import os
import pandas as pd

from src.classifier import classify_intent_batch
from src.clusterer import run_clustering_pipeline
from src.fallback import apply_smart_fallback

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("🚨 OPENAI_API_KEY not found in terminal environment!")

    print("⚙️ [FIT MODE] Starting Business-Steered Clustering Calibration...")
    
    print("📦 Loading historical training data...")
    df = pd.read_csv("data/mock_customer_data.csv")

    print("🧠 [Layer 1] Running LLM Intent Classification...")
    df_layer1 = df[['topic_key']].copy()
    df_layer1['intent_label'] = classify_intent_batch(df['customer_text'])

    print("📊 [Layer 2] Fitting algorithms and saving to models/ folder...")
    df_layer2 = run_clustering_pipeline(df, text_col="customer_text")

    print("🔗 [Layer 3] Applying Smart Fallback...")
    final_df = apply_smart_fallback(df_layer2, df_layer1)

    print("\n✅ Calibration Complete! Models successfully saved to disk.")
    print(final_df[["topic_key", "customer_text", "intent_label", "sub_topic", "final_main_topic"]].head(5).to_string(index=False))

if __name__ == "__main__":
    main()