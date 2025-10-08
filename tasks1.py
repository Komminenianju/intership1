# ----------------------
# 1. Import Libraries
# ----------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

# ----------------------
# 2. Extract (Load Raw Data)
# ----------------------
def extract_data(file_path: str) -> pd.DataFrame:
    """
    Reads data from CSV file
    """
    print("Extracting data...")
    return pd.read_csv(file_path)

# ----------------------
# 3. Transform (Clean + Preprocess)
# ----------------------
def transform_data(df: pd.DataFrame):
    """
    Cleans, imputes missing values, encodes categorical columns,
    and scales numeric features using a pipeline.
    """
    print("Transforming data...")

    # Separate features and target (assuming target column = 'target')
    X = df.drop(columns=["target"], errors="ignore")
    y = df["target"] if "target" in df.columns else None

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

    # Pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # Final Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])

    X_transformed = pipeline.fit_transform(X)

    return X_transformed, y, pipeline

# ----------------------
# 4. Load (Save Processed Data)
# ----------------------
def load_data(X, y, output_dir="processed_data"):
    """
    Saves transformed features and target as CSV files.
    """
    print("Loading (saving) processed data...")
    os.makedirs(output_dir, exist_ok=True)

    # Save features
    X_df = pd.DataFrame(X.toarray() if hasattr(X, "toarray") else X)
    X_df.to_csv(os.path.join(output_dir, "features.csv"), index=False)

    # Save target if available
    if y is not None:
        y.to_csv(os.path.join(output_dir, "target.csv"), index=False)

    print(f"Data saved to {output_dir}/")

# ----------------------
# 5. Main Execution
# ----------------------
if __name__ == "__main__":
    # Example usage (replace with your CSV file path)
    raw_data_path = "sample_data.csv"   # <-- Replace with actual dataset
    df = extract_data(raw_data_path)

    X, y, pipeline = transform_data(df)
    load_data(X, y)

    print("✅ Data pipeline executed successfully!")
