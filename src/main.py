"""
Main Entry Point for the Physician Notetaker AI System.
Coordinates the NLP pipeline from transcript to structured SOAP notes with full auditability.
"""

import argparse
import json
import os
import sys
import time
import logging
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PhysicianNotetaker")

# Ensure the src directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ner import MedicalNER
from summarizer import MedicalSummarizer
from sentiment import SentimentIntentAnalyzer
from soap_generator import SOAPGenerator

class PhysicianNotetakerPipeline:
    """
    Orchestrates the medical NLP pipeline, ensuring modularity and auditability.
    Standardizes communication between modules and enforces schema consistency.
    """

    def __init__(self):
        """
        Initializes the pipeline by loading all necessary modules.
        Models are loaded once via the shared ModelLoader (Singleton).
        """
        logger.info("Initializing Physician Notetaker AI System...")
        start_time = time.time()
        
        try:
            self.ner = MedicalNER()
            self.summarizer = MedicalSummarizer()
            self.sentiment_analyzer = SentimentIntentAnalyzer()
            self.soap_gen = SOAPGenerator()
            
            elapsed = time.time() - start_time
            logger.info(f"Pipeline initialized successfully in {elapsed:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

    def run(self, transcript: str, output_filename: str = "audit_output.json") -> Dict[str, Any]:
        """
        Processes a transcript through the complete AI pipeline.
        
        Args:
            transcript: Raw text of the doctor-patient conversation.
            output_filename: Name of the file to save results.
            
        Returns:
            Structured JSON output containing all analysis stages.
        """
        if not transcript or not transcript.strip():
            logger.error("Empty transcript provided.")
            return {"error": "Empty transcript provided", "status": "failed"}

        logger.info(f"Processing transcript (Length: {len(transcript)} characters)...")
        start_time = time.time()

        try:
            # 1. NER & Keywords
            ner_results = self.ner.process(transcript)
            
            # 2. Summarization
            summary_results = self.summarizer.summarize(transcript)
            
            # Enrich summary with structured NER data for the "Summary" view
            entities = ner_results.get("entities", {})
            if entities:
                summary_results.update({
                    "symptoms": [s['value'] for s in entities.get('Symptoms', [])],
                    "diagnosis": [d['value'] for d in entities.get('Diagnosis', [])],
                    "treatment": [t['value'] for t in entities.get('Treatment', [])],
                    "prognosis": [p['value'] for p in entities.get('Prognosis', [])]
                })

            # 3. Sentiment & Intent
            sentiment_intent = self.sentiment_analyzer.analyze(transcript)
            
            # 4. SOAP Note Generation
            soap_note = self.soap_gen.generate(transcript, ner_results, summary_results)

            # 5. Compile Final Output with Audit Metadata
            final_output = {
                "metadata": {
                    "system": "Physician Notetaker AI v1.1",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "processing_time_sec": round(time.time() - start_time, 4),
                    "transcript_length": len(transcript),
                    "auditability": "Confidence scores and source attribution included for all AI-generated fields.",
                    "device": "CPU (Offline-friendly)"
                },
                "clinical_summary": summary_results,
                "medical_entities": ner_results,
                "sentiment_and_intent": sentiment_intent,
                "soap_note": soap_note,
                "professional_notes": {
                    "confidence_audit": "Rule-based NER provides high precision for clinical terms. BART-based summarization and Zero-shot classification provide high-recall insights.",
                    "extensibility": "The pipeline uses standardized JSON interfaces, making it compatible with FHIR/HL7 standards for future EHR integrations."
                }
            }

            self._save_results(final_output, output_filename)
            logger.info(f"Processing complete in {final_output['metadata']['processing_time_sec']}s.")
            
            return final_output

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {
                "error": f"Internal pipeline error: {str(e)}",
                "status": "failed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

    def _save_results(self, data: Dict[str, Any], filename: str):
        """
        Saves output to the 'outputs' directory.
        
        Args:
            data: The JSON-serializable dictionary to save.
            filename: The target filename.
        """
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outputs_dir = os.path.join(root_dir, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        
        file_path = os.path.join(outputs_dir, filename)
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Results successfully saved to: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    """CLI entry point for the Physician Notetaker system."""
    parser = argparse.ArgumentParser(description="Physician Notetaker AI: Clinical NLP Pipeline")
    parser.add_argument("--text", type=str, help="Raw transcript text")
    parser.add_argument("--file", type=str, help="Path to transcript text file")
    parser.add_argument("--output", type=str, default="audit_output.json", help="Output filename")
    
    args = parser.parse_args()
    
    try:
        pipeline = PhysicianNotetakerPipeline()
    except Exception:
        sys.exit(1)
    
    transcript = ""
    if args.text:
        transcript = args.text
    elif args.file:
        try:
            with open(args.file, "r") as f:
                transcript = f.read()
        except Exception as e:
            logger.error(f"Could not read file: {e}")
            sys.exit(1)
    else:
        # Load sample for demonstration
        sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_transcript.txt")
        if os.path.exists(sample_path):
            with open(sample_path, "r") as f:
                transcript = f.read()
            logger.info("Using default sample transcript for demonstration.")
        else:
            logger.error("No input provided and sample_transcript.txt not found.")
            sys.exit(1)

    pipeline.run(transcript, args.output)

if __name__ == "__main__":
    main()
