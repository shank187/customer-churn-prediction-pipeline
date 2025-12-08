import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_model():
    print("Loading data...")
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Define Pipeline (Using your BEST hyperparameters found in the notebook)
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    
    num_transformer = SimpleImputer(strategy="median")
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    # The Winner Model
    model = XGBClassifier(
        n_estimators=415,
        learning_rate=0.1,
        max_depth=3,
        scale_pos_weight=2.76,
        subsample=0.9,
        n_jobs=-1,
        random_state=42
    )
    
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    
    # Train
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Validate
    score = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    print(f"Model trained. Validation ROC-AUC: {score:.4f}")
    
    # Save
    joblib.dump(pipeline, 'churn_model.joblib')
    print("Model saved to churn_model.joblib")

if __name__ == "__main__":
    train_model()