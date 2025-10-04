<<<<<<< HEAD
# Resume Screening Classifier

This repository contains a small demo `Resume_screen.py` that trains a toy RandomForest model
and exposes a Gradio UI to upload a resume (.pdf/.docx/.txt) and get a predicted job category.

Quick setup (PowerShell on Windows):

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
# If you plan to use spaCy's English model, also run:
python -m spacy download en_core_web_sm
```

3. Run the app:

```powershell
python "Resume_screen.py"
```

The Gradio UI will open in your browser. Upload a resume file and see the predicted category.

Notes:
- The demo uses a very small in-memory dataset. Replace with your labeled data and persist the model for real use.
- If spaCy isn't installed or its model isn't downloaded, the script falls back to a basic tokenizer.
=======
# Resume_Screening_Classifier
>>>>>>> c7d8c6f6df718dd984f246b79d3d46beeeebddc8
