import pandas as pd
import numpy as np
from src.utils.config_loader import load_config

config = load_config()
ANCHORS_CONFIG = config["anchors_config"]

def get_base_main_topic(sub_topic_name):
    """Helper to reverse-lookup the main category from the YAML."""
    for main_cat, sub_list in ANCHORS_CONFIG.items():
        if sub_topic_name in sub_list:
            return main_cat
    return "Other"

def apply_smart_fallback(df_layer2: pd.DataFrame, df_intent_layer1: pd.DataFrame) -> pd.DataFrame:
    """Merges Layer 1 and Layer 2 using pure Pandas."""
    
    # 1. Join the two dataframes on the unique ticket ID
    joined_df = pd.merge(df_layer2, df_intent_layer1, on="topic_key", how="left")
    
    # 2. Map the base main topic from the YAML config
    joined_df["base_main_topic"] = joined_df["sub_topic"].apply(get_base_main_topic)
    
    # 3. Apply the Smart Fallback Logic using vectorized NumPy
    joined_df["final_main_topic"] = np.where(
        joined_df["sub_topic"] == "Outlier/Noise",
        joined_df["intent_label"],     # If noise, fallback to LLM Intent
        joined_df["base_main_topic"]   # Otherwise, trust the mathematical cluster
    )
    
    # Clean up the temporary column
    joined_df = joined_df.drop(columns=["base_main_topic"])
    
    return joined_df