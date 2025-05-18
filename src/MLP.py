import pandas as pd
import argparse
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, get_model, get_xgboost_param_grid, plot_xgboost_importance_with_names

def run_pipeline(models, test_size, random_state):
    # Load data
    df = load_data()

    categorical_features = [
        "Age",
        "Occupation",
        "Marital Status",
        "Education Level",
        "Credit Default",
        "Housing Loan",
        "Personal Loan",
        "Contact Method",
        "Campaign Calls",
        "Previous Contact Days",
    ]

    ordinal_features = [
        "Age",
        "Education Level",
        "Campaign Calls",
        "Previous Contact Days",
    ]
    nominal_features = [
        "Occupation",
        "Marital Status",
        "Credit Default",
        "Housing Loan",
        "Personal Loan",
        "Contact Method",
    ]

    ordinal_ordering = [
        ["20s", "30s", "40s", "50s", "60s+"],  # Age
        [
            "illiterate",
            "unknown",
            "basic.4y",
            "basic.6y",
            "basic.9y",
            "high.school",
            "professional.course",
            "university.degree",
        ],  # Education Level
        ["1_call", "2_calls", "3-5_calls", "6-10_calls", "10+_calls"],  # Campaign Calls
        [
            "no_prev_contact",
            "last_5_days",
            "last_10_days",
            "last_15_days",
            "more_than_15_days",
        ],  # Previous Contact Days
    ]

    target = "Subscription Status"
    X = df[categorical_features]
    y = df[target].map({"yes": 1, "no": 0})

    print("\nClass distribution before split:")
    print(y.value_counts())

    # Train-test split
    print("[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Preprocessor
    ordinal_pipeline = Pipeline(
        [("ordinal", OrdinalEncoder(categories=ordinal_ordering))]
    )
    nominal_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(
        [
            ("ord", ordinal_pipeline, ordinal_features),
            ("nom", nominal_pipeline, nominal_features),
        ]
    )

    # Cross-validation
    print("\n[INFO] Performing 5-fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average="weighted")

    for name in models:
        print(f"\n[INFO] Training model: {name}")

        if name == "xgboost":
            print("[INFO] Performing GridSearchCV for XGBoost...")

            param_grid = get_xgboost_param_grid()

            pipe = ImbPipeline(
                [
                    ("preprocessor", preprocessor),
                    ("scaler", StandardScaler(with_mean=False)),
                    # ('SMOTETomek', SMOTETomek(random_state=42)),
                    (
                        "classifier",
                        XGBClassifier(eval_metric="logloss", random_state=42),
                    ),
                ]
            )

            grid_search = GridSearchCV(
                estimator=pipe,
                param_grid={f"classifier__{k}": v for k, v in param_grid.items()},
                scoring=f1_scorer,
                cv=cv,
                n_jobs=-1,
                verbose=1,
            )

            grid_search.fit(X_train, y_train)
            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"Best F1 Score: {grid_search.best_score_:.4f}")

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            print(
                f"\n[RESULT] Test Set Classification Report for {name} (after GridSearchCV):"
            )
            print(classification_report(y_test, y_pred))

            # Extract the XGBoost model from the pipeline
            xgb_model = best_model.named_steps["classifier"]

            # Extract the preprocessor from the pipeline
            preprocessor = best_model.named_steps["preprocessor"]

            feature_names, importances, indices = plot_xgboost_importance_with_names(
                xgb_model,
                preprocessor,
                ordinal_features,
                nominal_features,
                ordinal_ordering,
                figsize=(12, 8),
                save_path="xgboost_feature_importance.png",
            )

        else:
            model = get_model(name)
            pipe = ImbPipeline(
                [
                    ("preprocessor", preprocessor),
                    ("scaler", StandardScaler(with_mean=False)),
                    # ('SMOTETomek', SMOTETomek(random_state=42)),
                    ("classifier", model),
                ]
            )

            scores = cross_val_score(pipe, X_train, y_train, scoring="accuracy", cv=cv)
            print(
                f"[CV RESULT] {name} Accuracy: {scores.mean():.4f} ± {scores.std():.4f}"
            )

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            print(f"\n[RESULT] Test Set Classification Report for {name}:")
            print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP for Subscription Prediction")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "random_forest", "logistic_regression"],
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    run_pipeline(args.models, args.test_size, args.random_state)
