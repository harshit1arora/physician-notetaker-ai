# Physician Notetaker AI: Clinical NLP Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Production-Ready](https://img.shields.io/badge/Status-Production--Ready-green.svg)]()

## 🩺 Project Overview
**Physician Notetaker AI** is a production-quality clinical NLP pipeline designed to transform raw doctor-patient transcripts into structured, actionable medical documentation. It automates the generation of **SOAP notes** (Subjective, Objective, Assessment, Plan), extracts clinical entities with high precision, and analyzes patient sentiment and intent.

This project demonstrates a modular, extensible architecture suitable for integration with modern Electronic Health Record (EHR) systems.

---

## 🏗️ Architecture Diagram
```ascii
[ Raw Transcript ]
       |
       v
+-------------------------+
|   Main Pipeline Logic   |
| (PhysicianNotetaker)    |
+-----------+-------------+
            |
    +-------+-------+-----------------------+-----------------------+
    |               |                       |                       |
    v               v                       v                       v
+---------+   +------------+         +-------------+         +-------------+
| Medical |   |  Clinical  |         |  Sentiment  |         |    SOAP     |
|   NER   |   | Summarizer |         |  & Intent   |         |  Generator  |
+----+----+   +-----+------+         +------+------+         +------+------+
     |              |                       |                       |
     |              |        [ Shared Model Loader ]                |
     |              +-----------------------+-----------------------+
     |                                      |
     +------------------------------------->| [ Transformers / spaCy ]
                                            | [ CPU-Optimized Cache ]
                                            +-----------------------+
       |
       v
[ Structured JSON Output ]
(Audit Metadata + SOAP + Entities)
```

---

## 🚀 Design Philosophy
- **Modularity**: Each NLP task (NER, Sentiment, Summarization) is isolated in its own module with a standardized JSON interface.
- **Auditability & Confidence**: Every AI-generated field is accompanied by a confidence score and source attribution (e.g., `rule-based` vs `model-inference`).
- **Performance**: Heavy models are loaded once via a **Singleton Model Loader** and cached for the duration of the process.
- **Clinical Integrity**: SOAP mapping logic follows standard medical documentation practices, ensuring high readability for clinicians.

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### Installation
1. Clone the repository and navigate to the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Download the spaCy model if not present:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Running the System
Process a transcript via the CLI:
```bash
python src/main.py --file sample_transcript.txt --output results.json
```
Or use raw text directly:
```bash
python src/main.py --text "Patient reports fever and cough for 3 days."
```

---

## 📊 Models Used
- **Summarization**: `sshleifer/distilbart-cnn-12-6` (Lightweight, ~300MB, optimized for inference)
- **Zero-Shot Classification**: `facebook/bart-large-mnli` (Used for sentiment, intent, and fallback keyword detection)
- **NER**: spaCy `en_core_web_sm` + Custom `EntityRuler` (Optimized for clinical terms)
- **Execution**: Forced CPU-only mode for broad compatibility and lightweight runtime.

---

## 📝 Confidence & Auditability
We prioritize transparency in AI decision-making. The `metadata` block in every output includes:
- System version and timestamp.
- Total processing time.
- Attribution for every extracted entity.
- Confidence scores for classification tasks.

---

## 🧩 Extensibility
The pipeline is designed for future clinical NLP tasks:
- **FHIR/HL7 Compatibility**: JSON schemas are easily mappable to standard healthcare interoperability formats.
- **Plug-and-Play Modules**: New analyzers (e.g., ICD-10 coding, Risk Adjustment) can be added by implementing a simple `.process()` interface.
- **Multi-modal Support**: The architecture can be extended to accept audio inputs via Whisper/STT integration.

---

## ⚠️ Limitations & Ethical Disclaimer
- **Not a Diagnostic Tool**: This software is intended as a documentation assistant for licensed medical professionals. It does **not** provide medical advice or diagnosis.
- **Offline Mode**: While models run locally, initial setup requires an internet connection to download pre-trained weights.
- **Transcript Quality**: Accuracy is dependent on the quality and clarity of the input transcript.

---

## � Future Improvements
- [ ] Integration with medical-specific LLMs (e.g., Med-PaLM, BioGPT).
- [ ] Real-time streaming transcription processing.
- [ ] Support for multi-speaker identification (Diarization).
- [ ] Local vector store for long-term patient history context.
