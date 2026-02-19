import yaml
import os

def load_config(config_path="config/domain_config.yaml"):
    """Loads the YAML configuration file safely from anywhere in the project."""
    
    # This automatically finds your main project folder (sentence_similarity-semantic)
    # no matter where you run the code from!
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    full_path = os.path.join(base_dir, config_path)
    
    with open(full_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)