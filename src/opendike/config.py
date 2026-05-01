import yaml
import os
from typing import Any, Dict

class Config:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # Default path for config.yaml is the project root
        config_path = os.environ.get("OPENDIKE_CONFIG")
        
        if config_path is None:
            # Try to find config.yaml in CWD, then in the project root relative to this file
            if os.path.exists("config.yaml"):
                config_path = "config.yaml"
            else:
                # Fallback to project root (2 levels up from src/opendike/config.py)
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                config_path = os.path.join(base_dir, "config.yaml")

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            print(f"Warning: Configuration file not found at {config_path}. Using empty config.")

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

config = Config()
