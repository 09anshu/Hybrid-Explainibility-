# Hybrid Explainability for Chest X-Ray Analysis

A hybrid medical-imaging project that combines a deep learning chest X-ray classifier, Grad-CAM++ visual explanations, and an LLM-assisted radiology-style explanation workflow.

## Overview

This project is built around a **DenseNet121-based** chest X-ray pipeline for multi-label thoracic pathology prediction. It adds:
- **Grad-CAM++** heatmaps for visual explainability
- **Streamlit UI** for interactive inference
- **Gemini-powered explanation/chat layer** for follow-up discussion
- training, evaluation, and testing scripts for local experimentation

## Main Features

- Multi-label chest X-ray pathology prediction
- Explainability with Grad-CAM++ overlays
- Interactive web app using Streamlit
- Training and evaluation utilities
- Saved metrics and result plots under the `results/` folder
- Optional `.env`-based Gemini API integration

## Project Structure

- `app.py` — main Streamlit application
- `model.py` — model definition and label setup
- `dataset.py` — dataset and preprocessing utilities
- `train.py` — model training logic
- `run_local.py` — local training runner
- `evaluate.py` / `evaluate_phase3.py` — evaluation scripts
- `gradcam_utils.py` — Grad-CAM++ generation utilities
- `llm_explainer.py` — LLM explanation pipeline
- `test_pneumonia.py` — local test/inference helper
- `results/` — metrics, AUC report, and generated plots
- `implementation.md` — detailed implementation report
- `build_execution_log.md` — build and execution notes

## Repository Notes

This repository intentionally excludes very large or sensitive files, including:
- dataset folders such as `train/`, `valid/`, and `phase_3/`
- local virtual environments
- model checkpoints like `*.pth` / `*.pt`
- `.env` secrets

If you want full local execution, place the required datasets and trained weight files in the project root or expected folders.

## Setup

1. Create a virtual environment
2. Install the required dependencies
3. Add your API key in a `.env` file if you want chat/explanation support
4. Place trained model weights in the project root

### Example `.env`

```env
gemini_api_key=YOUR_GEMINI_API_KEY
```

## Suggested Dependencies

Install the core packages used by the app:

```bash
pip install torch torchvision streamlit pandas numpy matplotlib pillow python-dotenv langchain langchain-google-genai
```

Depending on your environment, you may also need Jupyter-related packages for the notebook.

## Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

## Training and Evaluation

Typical entry points:

```bash
python run_local.py --phase 1
python evaluate.py
python test_pneumonia.py
```

## Explainability Workflow

1. Upload a chest X-ray image
2. Run model inference
3. Generate Grad-CAM++ heatmaps
4. Review class probabilities and highlighted image regions
5. Use the LLM layer for a structured explanation or follow-up chat

## Results

Current result artifacts are stored in `results/`, including:
- `metrics.json`
- `training_history.json`
- `auc_report.txt`
- ROC / precision-recall / confusion-matrix plots

## Disclaimer

This project is for **research and educational use only**. It is **not** a medical device and must not be used as a substitute for professional clinical judgment.
