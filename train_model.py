"""Entraînement du modèle de credit scoring sur les données synthétiques.

Compare une régression logistique et une forêt aléatoire (validation croisée),
retient le meilleur modèle selon l'AUC ROC, puis le sauvegarde sous forme de
Pipeline scikit-learn complet (prétraitement + modèle) dans ``model/model.pkl``.

Usage :
    python generate_synthetic_data.py   # d'abord générer les données
    python train_model.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features

SEED = 42
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "donnees_synthetiques.xlsx"
MODEL_DIR = ROOT / "model"


def make_pipeline(estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",  # indicateurs de garanties + mois d'octroi
    )
    return Pipeline([("prep", preprocessor), ("clf", estimator)])


def main() -> None:
    df = pd.read_excel(DATA_PATH)
    X = build_features(df)
    y = df["STATUT"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    candidates = {
        "Régression logistique": make_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        ),
        "Forêt aléatoire": make_pipeline(
            RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=SEED, n_jobs=-1,
            )
        ),
    }

    best_name, best_model, best_auc = None, None, -1.0
    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
        print(f"{name} : AUC ROC (CV 5 plis) = {scores.mean():.3f} ± {scores.std():.3f}")
        if scores.mean() > best_auc:
            best_name, best_model, best_auc = name, pipe, scores.mean()

    best_model.fit(X_train, y_train)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    print(f"\nModèle retenu : {best_name}")
    print(f"AUC ROC (test) : {roc_auc_score(y_test, y_proba):.3f}")
    print(classification_report(y_test, y_pred, target_names=["Sain", "En défaut"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "model.pkl")
    print(f"Pipeline sauvegardé dans {MODEL_DIR / 'model.pkl'}")


if __name__ == "__main__":
    main()
