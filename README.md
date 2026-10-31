# Explainable Malware Detection System

**Assignment:** Malware Analysis Case Studies — Project 9, Explainable Malware Detection System.

A student-friendly prototype that trains and compares a Random Forest and Decision Tree on static PE-feature tables, evaluates them, predicts labels for uploaded CSV rows, displays global feature importance, and optionally uses Google Gemini to explain a selected result in plain language.

## Important dataset note

`data/demo_training_data.csv` is **synthetic demonstration data** generated for testing the app workflow. It is not a real malware dataset and must not be used to claim real-world detection performance. For an academic evaluation, replace it with an appropriately licensed, labelled dataset of extracted PE features and document its source and limitations.

## Features

- CSV upload for labelled training data
- Random Forest and Decision Tree training and comparison
- Accuracy, weighted precision, weighted recall, F1-score, classification report, and confusion matrix
- CSV upload for unlabeled feature rows and downloadable predictions
- Global feature importance for model transparency
- Optional Gemini explanation, grounded in model output and supplied feature values
- No executable is run by this application

## Setup (Windows PowerShell)

1. Install Python 3.10 or newer.
2. Extract this project folder and open PowerShell inside it.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, you can run the environment's Python directly:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

4. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
5. Start the dashboard:
   ```powershell
   python -m streamlit run app.py
   ```
6. Streamlit will print a local URL, usually `http://localhost:8501`.

## Enable Gemini explanations (optional)

1. Obtain an API key from Google AI Studio, subject to Google's availability and terms.
2. Copy `.env.example` to `.env`.
3. Replace the placeholder with your key:
   `GEMINI_API_KEY=your_actual_key`
4. Restart Streamlit if it was already running.

Never share or commit the real `.env` file. The app works for training, evaluation, and CSV predictions even without Gemini.

The app currently requests the `gemini-2.5-flash` model. If that model is unavailable for your API key or region, update the model identifier in `app.py` to one available to your account.

## CSV format

Training CSV must contain these numeric feature columns and a `label` column:

- `file_size`
- `num_sections`
- `num_imported_dlls`
- `num_imported_functions`
- `section_entropy`
- `has_writable_executable_section` (0 or 1)
- `virtual_raw_size_ratio`
- `unusual_section_name_count`
- `header_inconsistency_count`
- `label` (e.g., `benign` or `suspicious`)

Prediction CSV needs the nine feature columns; `label` is optional and ignored for prediction. Column names must match exactly.

## Optional command-line training

```powershell
python train_model.py
python train_model.py --csv path\to\your_labelled_features.csv
```

## Project workflow

1. Load a labelled feature dataset.
2. Split into training and held-out test subsets.
3. Train Random Forest and Decision Tree.
4. Compare metrics and inspect the confusion matrix.
5. Upload feature rows for prediction.
6. Inspect global feature importance.
7. Optionally ask Gemini for a cautious plain-language explanation.

## Scope and limitations

- This is a static-feature prototype, not a full malware sandbox.
- Feature importance is global; it is not a SHAP explanation of an individual prediction.
- Model scores are not guaranteed to be calibrated probabilities.
- Static features can be misleading, and a shared pattern does not prove maliciousness.
- Gemini explanations are generated text and can be wrong; verify claims against evidence.
- Do not upload confidential samples or sensitive data to third-party services without permission.
