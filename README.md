# AI Resume Job Matcher

An AI-powered Resume Job Matching web application that uses **Natural Language Processing (NLP)** and **Machine Learning** to evaluate how well a candidate's resume matches a given Job Description.

The application processes PDF/DOCX resumes, extracts text, converts resume and job-description text into TF-IDF features, calculates Cosine Similarity, checks domain compatibility, and generates a final Machine Learning based match prediction.

## Live Demo

https://ai-resume-job-matcher-acnj.onrender.com/

---

## Project Overview

Recruiters often need to compare a large number of resumes against job descriptions. Manually performing this comparison can be time-consuming.

This project automates the initial resume screening process by analyzing:

* Resume content
* Job description
* Text similarity
* Candidate and job domains
* Machine Learning classification probabilities

The system produces a final prediction such as:

* **Good Fit**
* **Potential Fit**
* **No Fit**

The application is built with **Python, NLP, Scikit-learn, Flask**, and deployed using **Render**.

---

## Key Features

* 📄 PDF resume processing
* 📝 DOCX resume processing
* 🧹 Text preprocessing and cleaning
* 🔤 TF-IDF Vectorization
* 📐 Cosine Similarity
* 🎯 Domain Matching
* 🤖 Machine Learning Classification
* 📊 Prediction probabilities
* ⚙️ Custom prediction threshold
* 🌐 Flask web application
* ☁️ Render deployment
* 💾 Saved ML model and TF-IDF vectorizer
* 📋 Structured prediction results

---

## How It Works

The application follows an end-to-end Machine Learning pipeline:

```text
Resume Upload
      │
      ▼
PDF / DOCX Text Extraction
      │
      ▼
Text Cleaning
      │
      ├───────────────┐
      ▼               ▼
Resume Text       Job Description
      │               │
      ▼               ▼
   TF-IDF          TF-IDF
      │               │
      └───────┬───────┘
              ▼
      Cosine Similarity
              │
              ▼
        Domain Matching
              │
              ▼
      Combined Features
              │
              ▼
     ML Classification
              │
              ▼
    Probability Analysis
              │
              ▼
     Custom Threshold
              │
              ▼
       Final Prediction
```

---

## NLP Techniques Used

### 1. Text Cleaning

The application performs basic text preprocessing:

* Converts text to lowercase
* Removes unnecessary whitespace
* Strips leading and trailing spaces

This ensures consistent input before vectorization.

### 2. TF-IDF Vectorization

TF-IDF stands for **Term Frequency-Inverse Document Frequency**.

It converts text into numerical feature vectors and gives greater importance to informative words.

The project uses the saved TF-IDF vectorizer during prediction so that new resumes and job descriptions are transformed using the same feature space used during model development.

### 3. Cosine Similarity

Cosine Similarity measures the similarity between the resume and job-description TF-IDF vectors.

A higher similarity score indicates that the textual content of the resume is more closely related to the job description.

### 4. Domain Matching

The application detects broad domains using predefined keywords.

Supported domains include:

* Software
* Data
* Finance
* Engineering
* Management

The detected resume domain and job-description domain are compared to generate an additional feature.

---

## Machine Learning Approach

The final prediction uses a trained Machine Learning classification model.

The model receives a combined feature representation containing:

1. Resume TF-IDF features
2. Job Description TF-IDF features
3. Cosine Similarity score
4. Domain Match feature

The application uses:

```python
model.predict_proba()
```

to obtain class probabilities.

A saved custom threshold is then applied to the **Good Fit** probability.

If the Good Fit probability reaches the saved threshold, the final prediction is:

```text
Good Fit
```

Otherwise, the application selects the strongest alternative class.

---

## Model Artifacts

The trained Machine Learning components are stored inside the `model/` directory.

```text
model/
├── resume_job_matching_model.pkl
├── tfidf_vectorizer.pkl
└── threshold.pkl
```

### `resume_job_matching_model.pkl`

Contains the trained Machine Learning classification model.

### `tfidf_vectorizer.pkl`

Contains the trained TF-IDF vectorizer used to transform resume and job-description text.

### `threshold.pkl`

Contains the optimized decision threshold used for the Good Fit prediction.

---

## Resume Processing

The application supports:

### PDF

PDF files are processed using:

```text
PyPDF
```

The application reads the available pages and extracts their text.

### DOCX

DOCX files are processed using:

```text
python-docx
```

The application extracts text from the document paragraphs.

---

## Tech Stack

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Joblib
* SciPy

### NLP

* TF-IDF Vectorization
* Cosine Similarity
* Text preprocessing

### Web Development

* Flask
* HTML
* CSS

### Document Processing

* PyPDF
* python-docx

### Deployment

* Render
* Gunicorn

### Development

* Jupyter Notebook

---

## Project Structure

```text
AI_Resume_Job_Matcher/
│
├── AI Resume Job Matcher/
│
├── JupyterNotebook/
│
├── model/
│   ├── resume_job_matching_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── threshold.pkl
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .gitattributes
└── .python-version
```

---

## Flask Application

The Flask backend provides two main routes.

### Home Route

```python
@app.route("/")
def home():
    return render_template("index.html")
```

This displays the resume upload and Job Description interface.

### Prediction Route

```python
@app.route("/predict", methods=["POST"])
def predict():
```

This route:

1. Receives the uploaded resume
2. Receives the Job Description
3. Extracts resume text
4. Detects domains
5. Performs NLP preprocessing
6. Calculates TF-IDF features
7. Calculates Cosine Similarity
8. Creates final ML features
9. Generates prediction probabilities
10. Applies the custom threshold
11. Displays the result

---

## Installation

Clone the repository:

```bash
git clone https://github.com/princeraj74/AI_Resume_Job_Matcher.git
```

Move into the project directory:

```bash
cd AI_Resume_Job_Matcher
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Locally

Start the Flask application:

```bash
python app.py
```

The application will run locally on the Flask development server.

Open the displayed local URL in your browser.

---

## How to Use

1. Open the application.
2. Upload a resume in PDF or DOCX format.
3. Paste the complete Job Description.
4. Click **Analyze Resume**.
5. The application extracts and processes the resume.
6. The system compares the resume with the Job Description.
7. The Machine Learning model generates prediction probabilities.
8. The final match result is displayed on the result page.

---

## Example Use Case

A student applying for a **Data Analyst** position can upload their resume and paste the Data Analyst Job Description.

The system analyzes:

```text
Resume
   +
Job Description
   ↓
NLP Processing
   ↓
TF-IDF Representation
   ↓
Cosine Similarity
   +
Domain Matching
   ↓
Machine Learning Model
   ↓
Final Match Prediction
```

This helps the candidate understand whether their resume is strongly aligned with the target role.

---

## Why This Project Is Different

This project does not rely only on keyword matching.

It combines multiple signals:

* Text representation
* Resume-JD similarity
* Domain compatibility
* Machine Learning classification
* Probability-based decision making
* Custom prediction threshold

This makes the project a practical demonstration of combining **NLP + Machine Learning + Web Development + Deployment**.

---

## Challenges Faced

### Consistent Preprocessing

The same text-cleaning logic needs to be used during both model development and production prediction.

### Feature Compatibility

The saved TF-IDF vectorizer and trained model must receive features in the expected format.

### Model Deployment

The trained ML artifacts need to be correctly loaded by the Flask application after deployment.

### Resume Extraction

PDF and DOCX files require different text-extraction approaches.

### Prediction Threshold

The default highest-probability class is not always the desired decision rule, so a custom threshold was implemented for the Good Fit class.

---

## Future Improvements

Some possible improvements include:

* Transformer-based NLP embeddings
* Semantic similarity using BERT or Sentence Transformers
* Better skill extraction
* Skill-gap analysis
* Resume scoring
* Job recommendation system
* Experience-level matching
* Education matching
* Improved domain classification
* Recruiter dashboard
* Resume ranking
* Larger and more diverse training dataset
* Model monitoring and performance tracking

---

## Limitations

The current domain detection system uses predefined keywords, so it may not correctly identify every possible professional domain.

The quality of the prediction also depends on the quality and representativeness of the training data.

The application is intended as a **portfolio and decision-support project** and should not be considered a replacement for professional recruitment judgment.

---

## Skills Demonstrated

This project demonstrates practical experience in:

* Python
* Machine Learning
* Natural Language Processing
* TF-IDF
* Cosine Similarity
* Feature Engineering
* Classification
* Probability-based prediction
* Flask
* Scikit-learn
* SciPy
* PDF/DOCX processing
* Model persistence with Joblib
* Web application development
* Deployment with Render

---

## Project Highlights

### Machine Learning

Built an end-to-end classification pipeline for resume-job matching.

### NLP

Applied text preprocessing, TF-IDF vectorization, and Cosine Similarity.

### Backend

Developed the prediction workflow using Flask.

### Deployment

Deployed the application as a live web service using Render.

---

## Author

**Prince Raj**

Aspiring Data Analyst | Python | SQL | Machine Learning | NLP | Data Analytics

GitHub: `princeraj74`

---

## Disclaimer

This project is developed for educational, portfolio, and demonstration purposes. The prediction should be considered an automated screening signal rather than a definitive hiring decision.

