# app/config/settings.py
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from app.utils.exceptions import ConfigurationError

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / 'config'

# Analysis settings
ANALYSIS_PROMPT_PATH = os.getenv('ANALYSIS_PROMPT_PATH') or str(CONFIG_DIR / 'prompts' / 'analysis_prompt.txt')
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', '1'))  # Delay between retries in seconds

# LLM Settings
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4')
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.7'))
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '2000'))

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# Category DB Path
CATEGORY_DB_PATH = os.getenv('CATEGORY_DB_PATH') or str(CONFIG_DIR / 'categories.json')


def validate_settings() -> None:
    """Validate critical settings and paths."""
    required_settings = {
        'ANALYSIS_PROMPT_PATH': ANALYSIS_PROMPT_PATH,
        'MAX_RETRIES': MAX_RETRIES,
        'LLM_MODEL': LLM_MODEL,
         'CATEGORY_DB_PATH': CATEGORY_DB_PATH,
    }

    for setting_name, setting_value in required_settings.items():
        if setting_value is None:
            raise ConfigurationError(
                f"Missing required setting: {setting_name}",
                details={'setting': setting_name}
            )

    # Validate prompt path exists
    prompt_path = Path(ANALYSIS_PROMPT_PATH)
    if not prompt_path.exists():
        raise ConfigurationError(
            f"Analysis prompt file not found: {ANALYSIS_PROMPT_PATH}",
            details={'path': str(prompt_path)}
        )
    # Validate category db path exists
    category_path = Path(CATEGORY_DB_PATH)
    if not category_path.exists():
        raise ConfigurationError(
            f"Category database file not found: {CATEGORY_DB_PATH}",
            details={'path': str(category_path)}
        )