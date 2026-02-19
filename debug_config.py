
import sys
import os

# Ensure the project root is in sys.path
sys.path.append(os.getcwd())

try:
    from src.utils.config_loader import load_config
    config = load_config()
    print("Config loaded successfully:", config)
except Exception as e:
    print("Error loading config:", e)
    import traceback
    traceback.print_exc()
