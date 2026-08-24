from pathlib import Path
import joblib


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "resume_job_matching_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.pkl"
THRESHOLD_PATH = BASE_DIR / "model" / "threshold.pkl"


# ---------------------------------------------------------
# Load saved ML files
# ---------------------------------------------------------

model = joblib.load(MODEL_PATH)

tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

best_threshold = joblib.load(THRESHOLD_PATH)


# ---------------------------------------------------------
# Verify loaded objects
# ---------------------------------------------------------

print("Saved model loaded successfully!")

print("\nModel type:")
print(type(model))

print("\nTF-IDF vectorizer loaded:")
print(type(tfidf_vectorizer))

print("\nSaved threshold:")
print(best_threshold)

print("\nModel classes:")
print(model.classes_)

print("\nTF-IDF feature count:")
print(tfidf_vectorizer.get_feature_names_out().shape[0])

print("\nModel input feature count:")
print(model.n_features_in_)
import re
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Same text cleaning logic used during model development
# ---------------------------------------------------------

def clean_text(text):
    text = str(text)

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


# ---------------------------------------------------------
# Resume + JD prediction
# ---------------------------------------------------------

def predict_resume_job_match(
    resume_text,
    jd_text,
    resume_domain,
    jd_domain
):

    # Clean resume and JD
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    # TF-IDF transformation
    resume_tfidf = tfidf_vectorizer.transform(
        [resume_clean]
    )

    jd_tfidf = tfidf_vectorizer.transform(
        [jd_clean]
    )

    # Text similarity
    similarity_score = cosine_similarity(
        resume_tfidf,
        jd_tfidf
    )[0][0]

    # Domain match
    domain_match = int(
        resume_domain.lower().strip()
        ==
        jd_domain.lower().strip()
    )

    # Combine features
    combined_features = __import__("scipy").sparse.hstack(
        [
            resume_tfidf,
            jd_tfidf
        ]
    )

    # Add similarity + domain match
    from scipy.sparse import csr_matrix

    extra_features = csr_matrix(
        [[
            similarity_score,
            domain_match
        ]]
    )

    final_features = __import__("scipy").sparse.hstack(
        [
            combined_features,
            extra_features
        ]
    )

    # Model probabilities
    probabilities = model.predict_proba(
        final_features
    )[0]

    # Normal prediction
    normal_prediction = model.classes_[
        probabilities.argmax()
    ]

    # Threshold prediction
    good_fit_index = list(
        model.classes_
    ).index("Good Fit")

    if probabilities[good_fit_index] >= best_threshold:

        threshold_prediction = "Good Fit"

    else:

        remaining_probabilities = probabilities.copy()

        remaining_probabilities[
            good_fit_index
        ] = -1

        threshold_prediction = model.classes_[
            remaining_probabilities.argmax()
        ]

    return {
        "prediction": threshold_prediction,
        "normal_prediction": normal_prediction,
        "probabilities": {
            label: float(probability)
            for label, probability
            in zip(model.classes_, probabilities)
        },
        "resume_domain": resume_domain,
        "jd_domain": jd_domain,
        "similarity_score": float(
            similarity_score
        ),
        "domain_match": domain_match
    }
print("\nPrediction function created successfully!")
from flask import Flask, render_template, request
from pypdf import PdfReader
from docx import Document
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

    # ---------------------------------------------------------
# Resume file text extraction
# ---------------------------------------------------------

def extract_resume_text(file):

    filename = file.filename.lower()

    # PDF
    if filename.endswith(".pdf"):

        reader = PdfReader(file)

        pages_text = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages_text.append(text)

        return "\n".join(pages_text)

    # DOCX
    elif filename.endswith(".docx"):

        document = Document(file)

        paragraphs = []

        for paragraph in document.paragraphs:

            if paragraph.text.strip():
                paragraphs.append(paragraph.text)

        return "\n".join(paragraphs)

    else:

        raise ValueError(
            "Only PDF and DOCX files are supported."
        )


# ---------------------------------------------------------
# Simple domain detection
# ---------------------------------------------------------

def detect_domain(text):

    text = text.lower()

    domain_keywords = {

        "software": [
            "software",
            "developer",
            "programming",
            "python",
            "java",
            "javascript",
            "react",
            "node",
            "backend",
            "frontend",
            "full stack",
            "web development",
            "api",
            "github"
        ],

        "data": [
            "data scientist",
            "data analyst",
            "machine learning",
            "deep learning",
            "statistics",
            "sql",
            "pandas",
            "numpy",
            "tensorflow",
            "pytorch",
            "data analysis",
            "data visualization",
            "power bi",
            "tableau"
        ],

        "finance": [
            "finance",
            "accounting",
            "accountant",
            "financial",
            "auditing",
            "audit",
            "banking",
            "investment",
            "portfolio",
            "tax",
            "budget",
            "ledger",
            "bookkeeping",
            "financial reporting"
        ],

        "engineering": [
            "engineer",
            "engineering",
            "mechanical",
            "civil",
            "electrical",
            "electronics",
            "manufacturing",
            "autocad",
            "solidworks",
            "design engineer",
            "production"
        ],

        "management": [
            "manager",
            "management",
            "project manager",
            "operations",
            "business development",
            "marketing",
            "sales",
            "leadership",
            "strategy",
            "human resources",
            "hr"
        ]
    }

    scores = {}

    for domain, keywords in domain_keywords.items():

        score = 0

        for keyword in keywords:

            if keyword in text:
                score += 1

        scores[domain] = score

    detected_domain = max(
        scores,
        key=scores.get
    )

    # If no domain keyword is found
    if scores[detected_domain] == 0:
        return "unknown"

    return detected_domain


# ---------------------------------------------------------
# Prediction route
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get uploaded resume
        resume_file = request.files.get("resume")

        # Get job description
        job_description = request.form.get(
            "job_description",
            ""
        ).strip()

        # Validate resume
        if not resume_file or resume_file.filename == "":

            return "Please upload a resume.", 400

        # Validate JD
        if not job_description:

            return "Please enter a job description.", 400

        # Extract resume text
        resume_text = extract_resume_text(
            resume_file
        )

        # Validate extracted text
        if not resume_text.strip():

            return (
                "Could not extract text from the resume.",
                400
            )

        # Detect domains
        resume_domain = detect_domain(
            resume_text
        )

        jd_domain = detect_domain(
            job_description
        )

        # Run ML prediction
        result = predict_resume_job_match(

            resume_text=resume_text,

            jd_text=job_description,

            resume_domain=resume_domain,

            jd_domain=jd_domain
        )

        # Send result to result page
        return render_template(
            "result.html",
            result=result
        )

    except Exception as e:

        print("\nPrediction Error:")
        print(e)

        return (
            f"Prediction failed: {str(e)}",
            500
        )
if __name__ == "__main__":
    app.run(debug=True)       