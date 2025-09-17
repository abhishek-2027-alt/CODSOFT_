import pandas as pd
import numpy as np
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data(zip_path, csv_name):
    """Load dataset from a zip file."""
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(f, encoding="ISO-8859-1", low_memory=False)
    print("✅ Dataset loaded successfully")
    return df


def preprocess_data(df):
    """Prepare features and target for model training."""
    print("Dataset shape:", df.shape)
    print("Available columns:", df.columns.tolist())

    # Target column
    target = "Rating"
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found.")

    # Drop rows where Rating is missing
    df = df.dropna(subset=[target])
    df[target] = pd.to_numeric(df[target], errors="coerce")

    # Select only relevant features
    features = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
    X = df[features]
    y = df[target]

    print("Features used:", features)
    print("Target:", target)

    # Preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[("cat", categorical_transformer, features)]
    )

    return X, y, preprocessor


def train_and_evaluate(X, y, preprocessor):
    """Train and evaluate a regression model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n📊 Model Evaluation:")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))


def main():
    zip_path = "IMDb Movies India.zip"   # update if needed
    csv_name = "IMDb Movies India.csv"  # update if different

    df = load_data(zip_path, csv_name)
    X, y, preprocessor = preprocess_data(df)
    train_and_evaluate(X, y, preprocessor)


if __name__ == "__main__":
    main()
