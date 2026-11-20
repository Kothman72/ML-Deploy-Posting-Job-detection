# Fake Job Posting Detection

An educational machine-learning project that analyzes job-posting text and
structured details to estimate whether a posting is fraudulent. It includes a
notebook for data exploration and training, a scikit-learn model pipeline, a
small Streamlit interface, and a command-line training script.

This is a learning and screening aid, not a definitive fraud detector. Review
each posting independently; model errors are possible.

## Dataset

The notebook is designed for the
[Fake Job Posting Detection Kaggle competition](https://www.kaggle.com/competitions/fake-job-posting).
The dataset is not included in this repository. Follow Kaggle's access and data
use terms, then either:

- Attach the competition data as notebook input when running on Kaggle.
- Download and extract `job_postings_train.csv` for local use.

The notebook looks for the training CSV in its working directory, the user's
Downloads directory, or Kaggle's competition input directory. The command-line
script also supports a `data` subdirectory and explicit file paths.

## Project files

- `notebook-fake-job-posting-detection.ipynb` — exploration, validation, model
  training, metrics, and model export.
- `app.py` — Streamlit form for scoring one job posting.
- `solve_fake_job_postings.py` — command-line training and optional test-set
  prediction.
- `requirements.txt` — Python dependencies.
- `model/` — created by the notebook when it saves the trained model.

## Setup

Create and activate a virtual environment from the project directory.

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow is only used by the notebook's neural-network examples; the saved
scikit-learn pipeline and Streamlit app use scikit-learn. TensorFlow availability
depends on the Python version and operating system.

## Train and evaluate

Open the notebook in Kaggle or in VS Code/Jupyter. Make sure the training data
is available, run the data-loading cell, then run the cell titled
**“Leak-safe word + character TF-IDF model.”**

The pipeline uses:

1. Word- and character-level TF-IDF features from job-posting text.
2. One-hot encoding for categorical details and imputation/scaling for numeric
   features.
3. Stratified cross-validation to compare Logistic Regression regularization
   and class-weight settings.
4. A validation split to select the fraud threshold, followed by one evaluation
   on a separate holdout split.

The notebook reports accuracy, precision, recall, F1, a confusion matrix, and a
fit-versus-cross-validation F1 gap. The gap is a diagnostic for possible
overfitting, not a guarantee that overfitting or underfitting is absent.
Because fraudulent posts are less common, accuracy can look high even when
fraud recall is low. Threshold changes can trade precision against recall, so
no single threshold is guaranteed to improve every metric.

After evaluation, the notebook refits the selected pipeline on all labeled
training rows and writes:

```text
model/fake_job_posting_pipeline.joblib
```

The file contains the fitted pipeline, decision threshold, and feature-column
metadata. The saved model is generated locally when the training cell runs; it
is not included in this repository.

## Run the Streamlit app

Place the generated `model` directory beside `app.py`. If training on Kaggle,
download the `model` directory from the notebook's working output and copy it
into the project directory. Then run:

```bash
streamlit run app.py
```

Enter a posting's text and structured details, then select **Check posting**.
The app displays the estimated fraud probability and applies the threshold
saved with the model. If the model file is missing, the app displays setup
instructions instead of attempting a prediction.

Only load model files you trust: Joblib model files use Python serialization.

## Command-line alternative

The standalone script can train and evaluate using local CSV files:

```bash
python solve_fake_job_postings.py --train path/to/job_postings_train.csv
```

If a competition test file is available, pass it to create probability
predictions:

```bash
python solve_fake_job_postings.py \
  --train path/to/job_postings_train.csv \
  --test path/to/job_postings_test.csv \
  --output submission.csv
```

## License

This repository includes the GNU General Public License, version 3. See
[`LICENSE`](./LICENSE) for details.
