import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def preprocess_network_flow_data(file_path, save_path):
    df = pd.read_csv(file_path)

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    pipeline = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    processed_data = pipeline.fit_transform(df)
    joblib.dump(pipeline, f"{save_path}/network_flow_pipeline.pkl")

    return processed_data
