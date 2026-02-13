"""
Summarization module for medical transcripts.
Generates concise, structured clinical summaries from doctor-patient conversations.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional
from model_loader import model_manager

class MedicalSummarizer:
    """
    Handles cleaning and summarization of medical transcripts using BART.
    Produces structured clinical summaries optimized for physician review.
    """

    def __init__(self):
        """
        Initializes the MedicalSummarizer with a shared BART summarization model.
        """
        self.summarizer = model_manager.get_summarizer()

    def clean_transcript(self, text: str) -> str:
        """
        Normalizes transcript text by removing excessive whitespace and speaker tags.
        
        Args:
            text: Raw transcript text.
            
        Returns:
            Cleaned and normalized text.
        """
        if not text:
            return ""
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Potential: remove timestamps or other artifacts if present
        return text

    def summarize(self, text: str) -> Dict[str, Any]:
        """
        Generates a clinical summary of the transcript.
        
        Args:
            text: Input transcript text.
            
        Returns:
            Structured summary dictionary with metadata and status.
        """
        if not text or not text.strip():
            return {
                "summary_text": "Empty transcript provided.", 
                "status": "empty_input",
                "metadata": {}
            }

        cleaned_text = self.clean_transcript(text)
        
        try:
            # BART max input is 1024 tokens. Truncation is handled by the pipeline.
            # max_length and min_length are tuned for clinical summaries
            summary = self.summarizer(
                cleaned_text, 
                max_length=150, 
                min_length=40, 
                do_sample=False,
                truncation=True
            )
            
            summary_text = summary[0]['summary_text']
            
            return {
                "patient_name": "Unknown",  # To be enriched by main pipeline
                "summary_text": summary_text,
                "symptoms": [],             # To be enriched by main pipeline
                "diagnosis": [],            # To be enriched by main pipeline
                "treatment": [],            # To be enriched by main pipeline
                "current_status": "Stable",
                "prognosis": "Good",
                "status": "success",
                "metadata": {
                    "model": "distilbart-cnn-12-6",
                    "input_length": len(cleaned_text),
                    "output_length": len(summary_text)
                }
            }
        except Exception as e:
            logging.error(f"Summarization failed: {e}")
            return {
                "summary_text": f"Summarization failed: {str(e)}",
                "status": "error",
                "metadata": {"error_detail": str(e)}
            }
