import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    db_path = "data/bmarket.db"
    connection = sqlite3.connect(db_path)
    query = "SELECT * FROM bank_marketing"
    raw_data = pd.read_sql_query(query, connection)
    connection.close()

    data_clean = raw_data.copy()

    # Convert 'Age' column in raw_data to integer by splitting the string
    data_clean["Age"] = data_clean["Age"].str.split().str[0].astype(int)

    # Remove rows where Age is 150
    data_clean = data_clean[data_clean["Age"] != "150 years"]

    # Remove rows where Campaign Calls is a negative number
    data_clean = data_clean[data_clean["Campaign Calls"] >= 0]

    # Remove Client ID column
    data_clean = data_clean.drop(columns=["Client ID"])

    # Impute "unknown" for missing values
    data_clean["Housing Loan"] = data_clean["Housing Loan"].fillna("unknown")
    data_clean["Personal Loan"] = data_clean["Personal Loan"].fillna("unknown")

    # Standardize 'Contact Method' values
    data_clean["Contact Method"] = data_clean["Contact Method"].replace(
        {
            "cellular": "Cell",
            "Cell": "Cell",
            "telephone": "Telephone",
            "Telephone": "Telephone",
        }
    )

    # Bin 'Previous Contact Days' into categories
    def categorize_pdays(p):
        if p == 999:
            return "no_prev_contact"
        elif p <= 5:
            return "last_5_days"
        elif p <= 10:
            return "last_10_days"
        elif p <= 15:
            return "last_15_days"
        else:
            return "more_than_15_days"

    data_clean["Previous Contact Days"] = data_clean["Previous Contact Days"].apply(
        categorize_pdays
    )

    # Bin 'Campaign Calls' into categories
    def bucket_campaign(x):
        if x == 1:
            return "1_call"
        elif x == 2:
            return "2_calls"
        elif 3 <= x <= 5:
            return "3-5_calls"
        elif 6 <= x <= 10:
            return "6-10_calls"
        else:
            return "10+_calls"

    data_clean["Campaign Calls"] = data_clean["Campaign Calls"].apply(bucket_campaign)

    # Bin 'Age' into categories
    def bucket_age(x):
        if x < 30:
            return "20s"
        elif x < 40:
            return "30s"
        elif x < 50:
            return "40s"
        elif x < 60:
            return "50s"
        else:
            return "60s+"

    data_clean["Age"] = data_clean["Age"].apply(bucket_age)

    return data_clean


def get_model(name):
    if name == "random_forest":
        return RandomForestClassifier(random_state=42)
    elif name == "xgboost":
        return XGBClassifier(eval_metric="logloss", random_state=42)
    elif name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unknown model: {name}")

def get_xgboost_param_grid():
    # Define parameter grid for XGBoost tuning
    param_grid = {
        "n_estimators": [75, 100, 150],  # Number of boosting rounds
        "learning_rate": [0.1],  # Step size shrinkage
        "max_depth": [3, 5, 7, 10],  # Depth of trees
        "subsample": [0.5, 0.7, 0.9],  # Fraction of training samples per tree
        "colsample_bytree": [0.5, 0.7, 0.9],  # Fraction of features per tree
    }
    return param_grid

# Add this function definition after get_xgboost_param_grid()
def plot_xgboost_importance_with_names(
    model,
    preprocessor,
    ordinal_features,
    nominal_features,
    ordinal_ordering,
    figsize=(12, 8),
    save_path=None,
):
    """
    Plot XGBoost feature importance with proper feature names after preprocessing

    Parameters:
    -----------
    model : XGBClassifier
        The XGBoost model extracted from pipeline
    preprocessor : ColumnTransformer
        The preprocessor from the pipeline
    ordinal_features : list
        List of ordinal feature names
    nominal_features : list
        List of nominal feature names
    ordinal_ordering : list of lists
        Ordered categories for ordinal features
    figsize : tuple
        Figure size for the plot
    save_path : str
        Path to save the plot, if None just displays it
    """

    importances = model.feature_importances_
    feature_names = []

    # Add ordinal feature names 
    for feature in ordinal_features:
        feature_names.append(feature)

    # Add one-hot encoded feature names
    for i, feature in enumerate(nominal_features):

        onehot = preprocessor.transformers_[1][1].named_steps["onehot"]
        categories = onehot.categories_[i]
        for category in categories:
            feature_names.append(f"{feature}_{category}")

    # Check if the number of feature names matches the number of importances
    if len(feature_names) != len(importances):
        print(
            f"Warning: Number of feature names ({len(feature_names)}) doesn't match "
            f"number of importance values ({len(importances)})"
        )
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    # Sort features by importance
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=figsize)
    plt.title("Feature Importance", fontsize=14)
    n_features = min(20, len(indices))

    # Create bar plot
    plt.bar(range(n_features), importances[indices[:n_features]], align="center")
    plt.xticks(
        range(n_features), [feature_names[i] for i in indices[:n_features]], rotation=90
    )
    plt.xlim([-1, n_features])
    plt.tight_layout()

    # Add values on top of bars
    for i, v in enumerate(importances[indices[:n_features]]):
        plt.text(i, v + 0.002, f"{v:.4f}", ha="center")

    if save_path:
        plt.savefig(save_path)
        print(f"Saved feature importance plot to {save_path}")

    plt.show()

    # Print top 10 features and their importance
    print("\nTop 10 features by importance:")
    for i in range(min(10, len(indices))):
        idx = indices[i]
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    return feature_names, importances, indices