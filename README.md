# Bank Marketing Subscription Prediction Pipeline

## Personal Information
**Full Name:** Chan Wei Wen Kevin |
**Email Address:** kevinchanfsa@gmail.com

## Project Overview
This project implements a machine learning pipeline for predicting subscription status from a bank marketing campaign dataset. The pipeline includes exploratory data analysis (EDA), data preprocessing, feature engineering, model training, and evaluation. Multiple classification models are compared to find the optimal approach for predicting whether a client will subscribe to a term deposit.

## Folder Structure
```
├── .github             
├── src                     
│   └── MLP.py                   # Model training and evaluation script
│   └── utils.py                 # Helper functions
├── eda.ipynb                    # Exploratory Data Analysis notebook
├── requirements.txt             # Python dependencies
├── run.sh                       # Executable bash script
└── README.md                    
```

## Execution Instructions

### Requirements
All dependencies are listed in `requirements.txt` and will be installed automatically during assessment.

**Python Version = 3.12**

### Running the Pipeline
1. Ensure that 'bmarket.db' is placed in 'data/bmarket.db'

2. To execute the ML pipeline, run:
```bash
./run.sh
```

This script will:
1. Run the MLP.py script to load data, train and evaluate the models

### Modifying Parameters
You can customize the model training by modifying the following parameters in MLP.py directly:

```bash
parser.add_argument('--models', nargs='+', default=["xgboost", "random_forest", "logistic_regression"])
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--random_state', type=int, default=42,)
```

Parameters:
- `--models`: List of models to train (options: xgboost, random_forest, logistic_regression)
- `--test_size`: Proportion of data to use for testing (default: 0.2)
- `--random_state`: Random seed for reproducibility (default: 42)

## Pipeline Flow
The machine learning pipeline follows these logical steps:

1. **Data Loading**: Loading data using SQLite
2. **Data Cleaning**: Handle missing values by imputation, removal of quantitative outliers
3. **Feature Engineering**: Create categorical bins for numerical features, encode categorical variables
4. **Data Splitting**: Create stratified train/test split
5. **Data Preprocessing**: Apply ordinal and one-hot encoding based on feature type
6. **Model Training**: Train multiple models with cross-validation
7. **Hyperparameter Tuning**: Optimize XGBoost parameters using GridSearchCV
8. **Model Evaluation**: Compare models using classification metrics

## Key EDA Findings

The exploratory data analysis revealed several important insights that informed our pipeline design:

1. **Age Distribution**: Client age has a significant impact on subscription likelihood, with clients in their 20s and 60s+ showing higher subscription rates.
2. **Contact Method**: Cellular contact had significantly higher success rates than telephone contact.
3. **Campaign Calls**: Diminishing returns observed after 5 calls, with negative impact after 10+ calls.
4. **Previous Contact**: Clients who have been contacted before have higher subscription likelihood
5. **Outliers**: Detected and removed instances with age=150 years and negative values in campaign calls.

Several feature engineering decisions were made:
1. Binned 'Age' into meaningful categories (20s, 30s, 40s, 50s, 60s+)
2. Binned 'Campaign Calls' into number of calls (1, 2, 3-5, 6-10, 10+)
3. Categorized 'Previous Contact Days', converted 999 into 'no_prev_contact' and days <=5 representing 'last_5_days'

## Feature Processing

| Feature | Data Type | Preprocessing | Engineering |
|---------|-----------|---------------|------------|
| Age | Numeric | Ordinal Encoding | Binned into categories: '20s', '30s', '40s', '50s', '60s+' |
| Occupation | Categorical | One-Hot Encoding | - |
| Marital Status | Categorical | One-Hot Encoding | - |
| Education Level | Categorical | Ordinal Encoding | Ordered by education level |
| Credit Default | Categorical | One-Hot Encoding | - |
| Housing Loan | Categorical | One-Hot Encoding | Missing values imputed as 'unknown' |
| Personal Loan | Categorical | One-Hot Encoding | Missing values imputed as 'unknown' |
| Contact Method | Categorical | One-Hot Encoding | Standardized values ('Cell', 'Telephone') |
| Campaign Calls | Numeric | Ordinal Encoding | Binned into categories: '1_call', '2_calls', '3-5_calls', '6-10_calls', '10+_calls' |
| Previous Contact Days | Numeric | Ordinal Encoding | Categorized: 'no_prev_contact', 'last_5_days', 'last_10_days', 'last_15_days', 'more_than_15_days' |

## Model Selection

Three classification models were selected for this pipeline:

1. **XGBoost Classifier**:
   - Chosen for its ability to handle imbalanced data and categorical features
   - Generally performs well on tabular data with complex relationships
   - Optimized using GridSearchCV to find optimal hyperparameters

2. **Random Forest Classifier**:
   - Robust to overfitting and handles non-linear relationships
   - Provides feature importance for model interpretability
   - Less sensitive to outliers than linear models

3. **Logistic Regression**:
   - Provides a baseline model for comparison
   - Performs well on linearly separable data

## Model Evaluation

The models were evaluated using the following metrics:

1. **Precision**
2. **Recall**
3. **F1-Score**

### Results Summary

**XGBoost** performed best with the following metrics on the test set:
- **Weighted Avg Precision:** 0.88
- **Weighted Avg Recall:** 0.90
- **Weighted Avg F1-score:** 0.87

While the precision and recall for for the non-subscription class 0 is high, the low precision and recall for the subscription class 1 indicates that the model is missing many potential subscribers and facing a class imbalance issue

### Limitations

1. **Targeted Marketing Efficiency**:  The severe imbalance in the dataset (11.4% subscription rate) remains a fundamental challenge, potentially limiting the model's ability to identify all potential subscribers

2. **Limited Context:** The model doesn't incorporate broader market conditions, interest rates, or competitive offerings that might influence customer decisions

3. **Channel Constraints:** While the model captures contact method, it doesn't account for emerging digital channels

## Business Value

This subscription prediction model offers business value to a bank's marketing department:

1. **Targeted Marketing Efficiency**: 
   - By accurately identifying clients more likely to subscribe to term deposits, the bank can focus marketing resources on high-probability prospects
   - Cost reduction by minimizing outreach to unlikely subscribers

2. **Customer Segmentation Insights**: 
   - The feature importance results reveal which client characteristics most influence subscription decisions
   - Creation of tailored messaging for different customer segments

3. **Campaign Optimization**: 
   - Based on the "Campaign Calls" feature importance, the bank can optimize the number of contact attempts
   - Identify the optimal contact time based on previous interaction patterns
   - Reduce customer annoyance from excessive contacts


