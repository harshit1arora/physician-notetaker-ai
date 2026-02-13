"""
Sentiment and Intent analysis module for medical transcripts.
Uses zero-shot classification to detect patient emotions and communicative intent.
"""

import json
import logging
from typing import Dict, Any, List
from model_loader import model_manager

class SentimentIntentAnalyzer:
    """
    Analyzes patient sentiment and intent using a zero-shot classification model.
    Provides structured JSON output with normalized confidence scores.
    """

    def __init__(self):
        """
        Initializes the analyzer with a shared zero-shot classification model.
        Pre-defines clinically relevant classes for zero-shot inference.
        """
        self.classifier = model_manager.get_classifier()
        
        # Define clinical-relevant classes
        self.sentiment_classes = ["Anxious", "Neutral", "Reassured", "Frustrated", "Hopeful"]
        self.intent_classes = [
            "Reporting Symptoms", 
            "Seeking Reassurance", 
            "Expressing Relief", 
            "Requesting Medication",
            "Inquiring about Prognosis"
        ]

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyzes the sentiment and intent of the provided text.
        
        Args:
            text: Input transcript text.
            
        Returns:
            Dictionary containing sentiment and intent labels with confidence scores.
        """
        if not text or not text.strip():
            return {
                "sentiment": {"label": "N/A", "confidence": 0.0},
                "intent": {"label": "N/A", "confidence": 0.0},
                "status": "empty_input"
            }

        try:
            # Perform zero-shot classification for sentiment
            sentiment_res = self.classifier(text, self.sentiment_classes)
            # Perform zero-shot classification for intent
            intent_res = self.classifier(text, self.intent_classes)
            
            return {
                "sentiment": {
                    "label": sentiment_res['labels'][0],
                    "confidence": round(float(sentiment_res['scores'][0]), 4)
                },
                "intent": {
                    "label": intent_res['labels'][0],
                    "confidence": round(float(intent_res['scores'][0]), 4)
                },
                "status": "success",
                "model": "bart-large-mnli"
            }
        except Exception as e:
            logging.error(f"Sentiment/Intent analysis failed: {e}")
            return {
                "error": f"Sentiment/Intent analysis failed: {str(e)}",
                "sentiment": {"label": "Error", "confidence": 0.0},
                "intent": {"label": "Error", "confidence": 0.0},
                "status": "error"
            }
