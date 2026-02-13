"""
Singleton Model Loader for the Physician Notetaker AI System.
Ensures that heavy NLP models are loaded into memory only once and shared across modules.
"""

import os
import spacy
import torch
from transformers import pipeline
from typing import Any, Dict, Optional

class ModelLoader:
    """
    Singleton class to manage and provide access to pre-trained NLP models.
    Ensures that heavy models are loaded only once to save memory and compute.
    """
    _instance = None
    _models: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def _get_device(self) -> int:
        """Determines the device to use for inference (CPU friendly)."""
        # Forced to CPU (-1) for stability in shared environments as requested
        return -1

    def get_spacy_model(self, model_name: str = "en_core_web_sm") -> Any:
        """
        Loads and returns the spaCy model. Fallback to basic regex if model missing.
        """
        if "spacy" not in self._models:
            print(f"[MODEL] Loading spaCy model: {model_name}...")
            try:
                self._models["spacy"] = spacy.load(model_name)
            except Exception as e:
                print(f"[MODEL] Could not load spaCy model {model_name}: {e}")
                print("[MODEL] Falling back to blank English model for basic NLP...")
                try:
                    self._models["spacy"] = spacy.blank("en")
                except:
                    self._models["spacy"] = None
        return self._models["spacy"]

    def get_summarizer(self, model_name: str = "sshleifer/distilbart-cnn-12-6") -> Any:
        """
        Loads and returns the BART summarization pipeline.
        
        Args:
            model_name: HuggingFace model identifier.
            
        Returns:
            A transformers pipeline for summarization.
        """
        if "summarizer" not in self._models:
            print(f"[MODEL] Loading summarization model: {model_name}...")
            self._models["summarizer"] = pipeline(
                "summarization", 
                model=model_name, 
                device=self._get_device()
            )
        return self._models["summarizer"]

    def get_classifier(self, model_name: str = "facebook/bart-large-mnli") -> Any:
        """
        Loads and returns the zero-shot classification pipeline.
        
        Args:
            model_name: HuggingFace model identifier.
            
        Returns:
            A transformers pipeline for zero-shot classification.
        """
        if "classifier" not in self._models:
            print(f"[MODEL] Loading zero-shot classification model: {model_name}...")
            self._models["classifier"] = pipeline(
                "zero-shot-classification", 
                model=model_name, 
                device=self._get_device()
            )
        return self._models["classifier"]

    def get_keyword_model(self) -> Any:
        """
        Loads and returns a keyword extraction pipeline.
        Uses a zero-shot classifier as a fallback if KeyBERT is unavailable.
        """
        if "keywords" not in self._models:
            print("[MODEL] Initializing keyword extraction (zero-shot fallback)...")
            self._models["keywords"] = self.get_classifier()
        return self._models["keywords"]

# Global instance for easy access
model_manager = ModelLoader()
