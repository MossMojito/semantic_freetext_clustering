import os
import pandas as pd
from openai import OpenAI
from src.utils.config_loader import load_config

config = load_config()
TARGET_INTENTS = config["target_intents"]

def classify_intent_batch(texts: pd.Series) -> pd.Series:
    """Classifies a Pandas Series of texts using the OpenAI API."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    prompt = f"""Classify the user text intent into exactly one of these categories:
    {TARGET_INTENTS}
    
    CRITICAL RULES:
    1. Output the category EXACTLY as it appears in the list.
    2. Do NOT change capitalization or grammar. 
    3. Return ONLY the category name. No explanations.
    """
    
    results = []
    for text in texts:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt}, 
                    {"role": "user", "content": f"User: \"{text}\"\nIntent:"}
                ],
                temperature=0.0
            )
            raw_text = str(response.choices[0].message.content).lower().strip()
            
            final_label = "Unidentified"
            for correct_intent in TARGET_INTENTS:
                if correct_intent.lower() in raw_text:
                    final_label = correct_intent 
                    break
                
            results.append(final_label)
        except Exception:
            results.append("Unidentified")
            
    return pd.Series(results, index=texts.index)