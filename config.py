"""
Configuration management for the competitive research multi-agent system.
Uses environment variables with fallback defaults.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")

# Model Configuration - Strictly GPT-5-mini as requested
OPENAI_MODEL: str = "gpt-5-mini"
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "1.0"))

# Tavily Search Configuration
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
TAVILY_SEARCH_DEPTH: str = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")  # "basic" or "advanced"
TAVILY_TOPIC: str = os.getenv("TAVILY_TOPIC", "general")  # "general" or "news"
TAVILY_INCLUDE_ANSWER: bool = os.getenv("TAVILY_INCLUDE_ANSWER", "true").lower() == "true"
TAVILY_INCLUDE_IMAGES: bool = os.getenv("TAVILY_INCLUDE_IMAGES", "false").lower() == "true"

# Request Configuration
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

# Agent Configuration
MAX_CONCURRENT_SEARCHES: int = int(os.getenv("MAX_CONCURRENT_SEARCHES", "5"))
RESEARCH_DEPTH: str = os.getenv("RESEARCH_DEPTH", "comprehensive")  # "basic", "standard", "comprehensive"

# Output Configuration
OUTPUT_FORMAT: str = os.getenv("OUTPUT_FORMAT", "markdown")  # "markdown", "json"
INCLUDE_CITATIONS: bool = os.getenv("INCLUDE_CITATIONS", "true").lower() == "true"
SAVE_RAW_DATA: bool = os.getenv("SAVE_RAW_DATA", "true").lower() == "true"

# Logging Configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Optional[str] = os.getenv("LOG_FILE")

def validate_config() -> bool:
    """Validate that required configuration is present."""
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required")
    
    if not TAVILY_API_KEY:
        errors.append("TAVILY_API_KEY is required")
    
    if OPENAI_MODEL != "gpt-5-mini":
        errors.append(f"Model must be gpt-5-mini, got: {OPENAI_MODEL}")
    
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

def get_config_summary() -> dict:
    """Get a summary of current configuration (without API keys)."""
    return {
        "model": OPENAI_MODEL,
        "temperature": TEMPERATURE,
        "max_search_results": MAX_SEARCH_RESULTS,
        "search_depth": TAVILY_SEARCH_DEPTH,
        "research_depth": RESEARCH_DEPTH,
        "output_format": OUTPUT_FORMAT,
        "include_citations": INCLUDE_CITATIONS,
        "api_keys_configured": {
            "openai": bool(OPENAI_API_KEY),
            "tavily": bool(TAVILY_API_KEY)
        }
    }