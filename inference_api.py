import os
import pandas as pd
from src.classifier import classify_intent_batch
from src.clusterer import predict_clustering_pipeline
from src.fallback import apply_smart_fallback

def run_interactive_inference():
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("🚨 OPENAI_API_KEY not found in terminal environment!")

    print("⚙️  Loading AI Models into memory (This takes a few seconds)...")
    
    # 1. Start the interactive loop
    print("\n" + "="*50)
    print("🚀 REAL-TIME AI TICKET ROUTING ENGINE ACTIVATED")
    print("Type 'exit' or 'quit' to close the engine.")
    print("="*50 + "\n")

    ticket_counter = 1

    while True:
        # 2. Wait for the user to type something in the terminal
        user_input = input("👤 Enter customer text: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("🛑 Shutting down Inference Engine. Goodbye!")
            break
        
        if not user_input:
            continue

        print("   🤖 Predicting...")

        # 3. Wrap the single input into a DataFrame format
        ticket_id = f"LIVE-{str(ticket_counter).zfill(3)}"
        df_new = pd.DataFrame([{"topic_key": ticket_id, "customer_text": user_input}])

        try:
            # 4. Push through the pipeline
            df_layer1 = df_new[['topic_key']].copy()
            df_layer1['intent_label'] = classify_intent_batch(df_new['customer_text'])
            
            df_layer2 = predict_clustering_pipeline(df_new, text_col="customer_text")
            
            final_df = apply_smart_fallback(df_layer2, df_layer1)
            
            # 5. Extract the final predictions
            intent = final_df['intent_label'].iloc[0]
            sub_topic = final_df['sub_topic'].iloc[0]
            main_topic = final_df['final_main_topic'].iloc[0]
            
            # 6. Print the result beautifully
            print(f"   🎯 Intent:       {intent.upper()}")
            print(f"   📂 Sub-Topic:    {sub_topic}")
            print(f"   🏢 Routed Dept:  {main_topic}")
            print("-" * 50)
            
            ticket_counter += 1

        except Exception as e:
            print(f"   ❌ Error processing text: {e}")

if __name__ == "__main__":
    run_interactive_inference()