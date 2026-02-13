"""
Named Entity Recognition (NER) module for medical transcripts.
Extracts clinical entities (Symptoms, Diagnosis, Treatment, Prognosis) and keywords.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from model_loader import model_manager

class MedicalNER:
    """
    Handles extraction of medical entities and keywords using spaCy and KeyBERT.
    Standardizes output into structured JSON with confidence scores.
    """

    def __init__(self):
        """
        Initializes the MedicalNER with a shared spaCy model and custom clinical patterns.
        """
        self.nlp = model_manager.get_spacy_model()
        
        # Ensure EntityRuler is added for pattern-based matching
        if self.nlp:
            try:
                if "entity_ruler" not in self.nlp.pipe_names:
                    self.ruler = self.nlp.add_pipe("entity_ruler", before="ner" if "ner" in self.nlp.pipe_names else None)
                else:
                    self.ruler = self.nlp.get_pipe("entity_ruler")
                self._initialize_patterns()
            except Exception as e:
                logging.warning(f"Could not initialize spaCy EntityRuler: {e}. Falling back to regex.")
                self.ruler = None
        else:
            self.ruler = None

        self.kw_model = model_manager.get_keyword_model()

    def _initialize_patterns(self):
        """
        Defines and adds clinical patterns to the spaCy EntityRuler.
        """
        patterns = [
            # Symptoms
            {"label": "SYMPTOM", "pattern": [{"LOWER": "fever"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "cough"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "headache"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "fatigue"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "nausea"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "pain"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "shortness"}, {"LOWER": "of"}, {"LOWER": "breath"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "chest"}, {"LOWER": "pain"}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": "hypertension"}]}, # Sometimes reported as symptom
            
            # Diagnosis
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "hypertension"}]},
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "diabetes"}]},
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "asthma"}]},
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "bronchitis"}]},
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "influenza"}]},
            {"label": "DIAGNOSIS", "pattern": [{"LOWER": "pneumonia"}]},
            
            # Treatment
            {"label": "TREATMENT", "pattern": [{"LOWER": "aspirin"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "ibuprofen"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "antibiotics"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "paracetamol"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "rest"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "fluids"}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": "inhaler"}]},
            
            # Prognosis
            {"label": "PROGNOSIS", "pattern": [{"LOWER": "good"}]},
            {"label": "PROGNOSIS", "pattern": [{"LOWER": "stable"}]},
            {"label": "PROGNOSIS", "pattern": [{"LOWER": "improving"}]},
            {"label": "PROGNOSIS", "pattern": [{"LOWER": "guarded"}]},
            {"label": "PROGNOSIS", "pattern": [{"LOWER": "chronic"}]}
        ]
        self.ruler.add_patterns(patterns)

    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extracts medical entities from text using spaCy EntityRuler or simple regex fallback.
        
        Args:
            text: Input transcript text.
            
        Returns:
            Dictionary of categorized entities with confidence scores.
        """
        if not text or not text.strip():
            return {"Symptoms": [], "Diagnosis": [], "Treatment": [], "Prognosis": []}

        results = {
            "Symptoms": [],
            "Diagnosis": [],
            "Treatment": [],
            "Prognosis": []
        }

        # If spaCy is available, use it
        if self.nlp:
            doc = self.nlp(text)
            mapping = {
                "SYMPTOM": "Symptoms",
                "DIAGNOSIS": "Diagnosis",
                "TREATMENT": "Treatment",
                "PROGNOSIS": "Prognosis"
            }

            seen = set()
            for ent in doc.ents:
                if ent.label_ in mapping:
                    val_lower = ent.text.lower()
                    if val_lower not in seen:
                        results[mapping[ent.label_]].append({
                            "value": ent.text,
                            "confidence": 0.98,
                            "source": "rule-based"
                        })
                        seen.add(val_lower)
            
            # If we found entities with spaCy, we return
            if any(results.values()):
                return results

        # Fallback to manual pattern matching if spaCy failed or found nothing
        # (This is more robust for restricted environments)
        text_lower = text.lower()
        patterns = {
            "Symptoms": ["fever", "cough", "headache", "fatigue", "nausea", "pain", "shortness of breath", "chest pain"],
            "Diagnosis": ["hypertension", "diabetes", "asthma", "bronchitis", "influenza", "pneumonia"],
            "Treatment": ["aspirin", "ibuprofen", "antibiotics", "paracetamol", "rest", "fluids", "inhaler"],
            "Prognosis": ["good", "stable", "improving", "guarded", "chronic"]
        }

        for category, terms in patterns.items():
            for term in terms:
                if term in text_lower:
                    # Find original casing if possible
                    import re
                    match = re.search(re.escape(term), text, re.IGNORECASE)
                    val = match.group(0) if match else term
                    
                    if not any(r['value'].lower() == term for r in results[category]):
                        results[category].append({
                            "value": val,
                            "confidence": 0.90,
                            "source": "regex-fallback"
                        })
        
        return results

    def extract_keywords(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts clinical keywords using a fallback zero-shot model or spaCy.
        
        Args:
            text: Input transcript text.
            
        Returns:
            List of keywords with confidence scores.
        """
        if not text or not text.strip():
            return []

        try:
            # Fallback keyword extraction using zero-shot classifier if KeyBERT is missing
            candidate_labels = ["symptoms", "medication", "follow-up", "diagnosis", "vitals"]
            res = self.kw_model(text, candidate_labels, multi_label=True)
            
            keywords = []
            for label, score in zip(res['labels'], res['scores']):
                if score > 0.3:
                    keywords.append({"keyword": label, "confidence": round(float(score), 4)})
            
            return keywords
        except Exception as e:
            logging.error(f"Keyword extraction failed: {e}")
            return []

    def process(self, text: str) -> Dict[str, Any]:
        """
        Full processing pipeline for the NER module.
        
        Args:
            text: Input transcript text.
            
        Returns:
            Structured JSON-ready dictionary.
        """
        if not text or not text.strip():
            return {
                "entities": {"Symptoms": [], "Diagnosis": [], "Treatment": [], "Prognosis": []},
                "keywords": [],
                "status": "empty_input"
            }

        try:
            return {
                "entities": self.extract_entities(text),
                "keywords": self.extract_keywords(text),
                "status": "success"
            }
        except Exception as e:
            logging.error(f"NER processing failed: {e}")
            return {
                "error": f"NER processing failed: {str(e)}", 
                "entities": {}, 
                "keywords": [],
                "status": "error"
            }
