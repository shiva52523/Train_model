# train_model.py

import os
import statistics
import numpy as np
import psycopg2
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import joblib

# Load environment variables
load_dotenv()

# --- DB connection ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "job_stability")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "Sp@495520!_")

FINAL_SCORE_NAMES = [
    "job_stability",
    "retention_risk",
    "long_term_association",
    "monetary_motivation",
    "cultural_fit",
]

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
PCA_PATH = os.path.join(MODEL_DIR, "pca.pkl")

EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


def fetch_past_candidates(limit=1000):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, raw_text, experience, certifications, projects,
                   job_stability_score, retention_risk_score,
                   long_term_association_score, monetary_motivation_score,
                   cultural_fit_score
            FROM candidates
            WHERE raw_text IS NOT NULL AND raw_text <> ''
            ORDER BY id DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
        return [dict(zip(colnames, row)) for row in rows]
    except Exception as e:
        print(f"[ERROR] Could not fetch past candidates: {e}")
        return []


def extract_features_from_text(raw_text: str, parsed: dict) -> np.ndarray:
    work_exps = parsed.get("experience", []) or []
    certifications = parsed.get("certifications", []) or []
    projects = parsed.get("projects", []) or []

    durations = [exp.get("duration_months", 0) for exp in work_exps if isinstance(exp, dict)]
    avg_duration = statistics.mean(durations) if durations else 0
    median_duration = statistics.median(durations) if durations else 0
    num_switches = max(0, len(work_exps) - 1)
    longest_job = max(durations) if durations else 0
    short_jobs_ratio = sum(1 for d in durations if d < 12) / len(work_exps) if work_exps else 0

    text_l = raw_text.lower()

    features = [
        avg_duration,
        median_duration,
        num_switches,
        longest_job,
        short_jobs_ratio,
        len(certifications),
        len(projects),
        int(any(w in text_l for w in ["salary", "ctc", "hike", "bonus", "compensation"])),
        int("team" in text_l or "collaborat" in text_l),
        int("remote" in text_l),
        int(any(w in text_l for w in ["volunteer", "community", "mentor", "open source"])),
        int("freelance" in text_l or "independent contractor" in text_l),
    ]

    if raw_text.strip():
        emb = EMB_MODEL.encode(raw_text)
        features.extend(emb.tolist())

    return np.array(features)


def safe_score(value, default=25.0):
    try:
        v = float(value)
        return max(0.0, min(100.0, v))
    except Exception:
        return default


def train_and_save_models():
    past_candidates = fetch_past_candidates()
    if not past_candidates:
        print("⚠️ No candidates found in DB. Creating fallback (dummy) models...")

        # Create dummy PCA
        dummy_matrix = np.random.rand(20, 384 + 13)  # 384 from embeddings + 13 handcrafted features
        # pca = PCA(n_components=50)
        max_components = min(50, dummy_matrix.shape[0], dummy_matrix.shape[1])
        if max_components < 2:  # PCA needs at least 2 components
            max_components = 2

        pca = PCA(n_components=max_components)
        pca.fit(dummy_matrix)
        joblib.dump(pca, PCA_PATH)
        print("✅ Saved fallback PCA model.")

        # Create dummy LightGBM models
        for score_name in FINAL_SCORE_NAMES:
            X_dummy = np.random.rand(50, 50)
            y_dummy = np.random.randint(20, 80, size=50)  # random scores 20–80
            model = LGBMRegressor()
            model.fit(X_dummy, y_dummy)
            joblib.dump(model, os.path.join(MODEL_DIR, f"{score_name}.pkl"))
            print(f"✅ Created fallback model for {score_name}.")
        return

    # If we have candidates in DB
    print(f"Training models on {len(past_candidates)} candidates...")

    embeddings = []
    for c in past_candidates:
        f_vec = extract_features_from_text(c["raw_text"], c)
        embeddings.append(f_vec)
    embeddings = np.array(embeddings)

    # PCA reduction
    pca = PCA(n_components=50)
    embeddings_reduced = pca.fit_transform(embeddings)
    joblib.dump(pca, PCA_PATH)
    print("✅ Saved PCA model.")

    # Train LightGBM models
    for score_name in FINAL_SCORE_NAMES:
        X, y = [], []
        for i, c in enumerate(past_candidates):
            val = c.get(f"{score_name}_score")
            if val not in (None, "Not enough data"):
                X.append(embeddings_reduced[i])
                y.append(safe_score(val))
        if len(y) > 10:
            model = LGBMRegressor()
            model.fit(np.array(X), np.array(y))
            joblib.dump(model, os.path.join(MODEL_DIR, f"{score_name}.pkl"))
            print(f"✅ Trained model for {score_name}.")
        else:
            print(f"⚠️ Not enough data to train {score_name}. Using dummy model.")
            X_dummy = np.random.rand(50, 50)
            y_dummy = np.random.randint(20, 80, size=50)
            model = LGBMRegressor()
            model.fit(X_dummy, y_dummy)
            joblib.dump(model, os.path.join(MODEL_DIR, f"{score_name}.pkl"))


if __name__ == "__main__":
    train_and_save_models()
