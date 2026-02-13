"""
SOAP Note Generation module.
Synthesizes clinical data into the standard Subjective, Objective, Assessment, and Plan format.
"""

import re
import logging
from typing import Dict, Any, List, Optional

class SOAPGenerator:
    """
    Generates structured medical SOAP notes from NLP pipeline outputs.
    Synthesizes Subjective, Objective, Assessment, and Plan sections with clinical logic.
    """

    def generate(self, transcript: str, ner_results: Dict[str, Any], summary_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Synthesizes various NLP outputs into a structured SOAP note.
        
        Args:
            transcript: Original transcript text.
            ner_results: Extracted medical entities.
            summary_results: Generated summary text and metadata.
            
        Returns:
            Dictionary with standardized SOAP sections.
        """
        # Ensure we have entities dictionary
        entities = ner_results.get("entities", {})
        
        # 1. Subjective (Patient's story, symptoms, and concerns)
        subjective = self._generate_subjective(entities, summary_results)
        
        # 2. Objective (Observable facts, vitals discussed)
        objective = self._generate_objective(transcript)
        
        # 3. Assessment (Clinical impression and diagnosis)
        assessment = self._generate_assessment(entities)
        
        # 4. Plan (Treatment, follow-up, and prognosis)
        plan = self._generate_plan(entities)
        
        return {
            "Subjective": subjective,
            "Objective": objective,
            "Assessment": assessment,
            "Plan": plan
        }

    def _generate_subjective(self, entities: Dict[str, Any], summary: Dict[str, Any]) -> str:
        """Constructs the Subjective section focusing on patient reports."""
        symptoms = [s['value'] for s in entities.get('Symptoms', [])]
        symptom_str = f"Patient reports the following symptoms: {', '.join(symptoms)}." if symptoms else "No acute symptoms specifically reported by the patient."
        
        summary_text = summary.get('summary_text', 'No clinical summary available.')
        return f"{symptom_str} {summary_text}"

    def _generate_objective(self, transcript: str) -> str:
        """Constructs the Objective section based on observable mentions in transcript."""
        transcript_lower = transcript.lower()
        findings = []
        
        # Check for vitals or physical exam mentions
        if any(x in transcript_lower for x in ["blood pressure", "bp", "hypertension"]):
            findings.append("Blood pressure discussed")
        if any(x in transcript_lower for x in ["fever", "temperature", "temp", "degrees"]):
            findings.append("Body temperature noted")
        if any(x in transcript_lower for x in ["heart rate", "pulse", "bpm"]):
            findings.append("Heart rate discussed")
        if any(x in transcript_lower for x in ["lungs", "breath", "respiratory", "breathing"]):
            findings.append("Respiratory status mentioned")
        
        # Attempt to find specific measurements (simple regex)
        bp_match = re.search(r'(\d{2,3}/\d{2,3})', transcript)
        if bp_match:
            findings.append(f"Recorded BP: {bp_match.group(1)}")

        if not findings:
            return "Physical examination findings not explicitly detailed in the transcript. Vitals discussed were within normal limits unless otherwise specified."
        
        return "Objective observations and vitals: " + "; ".join(findings) + "."

    def _generate_assessment(self, entities: Dict[str, Any]) -> str:
        """Constructs the Assessment section with clinical impressions."""
        diagnoses = [d['value'] for d in entities.get('Diagnosis', [])]
        if not diagnoses:
            return "Clinical impression: Assessment pending further evaluation and diagnostic results."
        
        diag_str = ", ".join(diagnoses)
        return f"Primary assessment and clinical impression: {diag_str}."

    def _generate_plan(self, entities: Dict[str, Any]) -> str:
        """Constructs the Plan section including treatments and follow-up."""
        treatments = [t['value'] for t in entities.get('Treatment', [])]
        prognosis = [p['value'] for p in entities.get('Prognosis', [])]
        
        treatment_str = f"Management Plan: {', '.join(treatments)}." if treatments else "Management Plan: Continue current monitoring; follow-up as needed."
        prognosis_str = f"Prognosis is considered {', '.join(prognosis)}." if prognosis else "Prognosis: Stable."
        
        return f"{treatment_str} {prognosis_str}"
