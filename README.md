# HireSense AI

HireSense is a local-first, multi-modal AI interview assessment platform designed to automate preliminary candidate screening. It leverages Natural Language Processing (NLP), local Large Language Models (LLMs), and Computer Vision to ensure fair, transparent, and scalable evaluations.

## 🚀 Current Progress (Implemented Modules)

The backend currently runs entirely locally with no third-party API dependencies:

* **Hybrid NLP Resume Parser:** Extracts text from PDFs using `PyMuPDF`, classifies sections using a mathematically balanced **SVM with TF-IDF**, and extracts technical skills using a custom **spaCy NER** pipeline.
* **Job Alignment Matcher:** An unsupervised vector space model utilizing **Cosine Similarity** to score a candidate's extracted skills against a target Job Description.
* **Local Question Generator:** Integrates with **Ollama (Mistral)** to autonomously generate dynamic, personalized technical interview questions based on the candidate's parsed skills.
* **Real-Time Visual Proctoring:** A lightweight computer vision module (`proctor_model.py`) using **OpenCV** and the **MediaPipe Tasks API**. It tracks the Eye Aspect Ratio (EAR) for blink detection and computes 3D head-pose estimation (with a 20-degree deadzone) to flag if a candidate looks away from the screen.

## 🛠️ Tech Stack (Current)
* **Python 3.x**
* **NLP & ML:** `spacy`, `scikit-learn`, `pandas`
* **Computer Vision:** `opencv-python`, `mediapipe`
* **Local LLM:** `ollama` (Mistral)
* **Document Processing:** `pymupdf`

## 🔮 Future Implementation Pipeline

As the project evolves into a fully interactive Multi-Agent platform, the following features are planned for development:

* **Multi-Agent LLM System:** Deploying specialized AI agents (Interviewer, Inspector, and Reviewer) to conduct the interview, evaluate answers for relevance, and act as a security guardrail against prompt injection.
* **Speech Recognition (ASR):** Integrating a lightweight speech-to-text model (e.g., Whisper/FasterWhisper) to transcribe candidate voice responses in real-time.
* **Facial Emotion Recognition:** Implementing a Hybrid CNN with handcrafted filters (e.g., Gabor) to evaluate candidate confidence and stress levels via facial expressions.
* **Full-Stack Integration:** Connecting the Python AI backend to a responsive UI/UX frontend and a database to manage session states and candidate evaluation metrics.

## ⚙️ How to Run Locally

1. **Activate the Virtual Environment & Install Dependencies:**
   ```bash
   pip install pandas spacy scikit-learn pymupdf joblib ollama opencv-python mediapipe numpy
   python -m spacy download en_core_web_sm
