# Databricks notebook source

# MAGIC %pip install -e ..
# MAGIC %restart_python

# COMMAND ----------
# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from marvel_characters.config import ProjectConfig
from marvel_characters import PROJECT_DIR
import pickle

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path=PROJECT_DIR / "project_config_marvel.yml")
filepath = PROJECT_DIR / "data" / "marvel_characters_dataset.csv"

# COMMAND ----------
# Load the data
df = pd.read_csv(filepath)
cat_features = config.cat_features
num_features = config.num_features
target = config.target
parameters = config.parameters

# COMMAND ----------
# Preprocess
df.rename(columns={"Height (m)": "Height"}, inplace=True)
df.rename(columns={"Weight (kg)": "Weight"}, inplace=True)

# Universe
df["Universe"] = df["Universe"].fillna("Unknown")
counts = df["Universe"].value_counts()
small_universes = counts[counts < 50].index
df["Universe"] = df["Universe"].replace(small_universes, "Other")

# Teams
df["Teams"] = df["Teams"].notna().astype("int")

# Origin
df["Origin"] = df["Origin"].fillna("Unknown")

# Identity
df["Identity"] = df["Identity"].fillna("Unknown")
df = df[df["Identity"].isin(["Public", "Secret", "Unknown"])]

# Gender
df["Gender"] = df["Gender"].fillna("Unknown")
df["Gender"] = df["Gender"].where(df["Gender"].isin(["Male", "Female"]), other="Other")

# Marital status
df.rename(columns={"Marital Status": "Marital_Status"}, inplace=True)
df["Marital_Status"] = df["Marital_Status"].fillna("Unknown")
df["Marital_Status"] = df["Marital_Status"].replace("Widow", "Widowed")
df = df[df["Marital_Status"].isin(["Single", "Married", "Widowed", "Engaged", "Unknown"])]

# Magic
df["Magic"] = df["Origin"].str.lower().apply(lambda x: int("magic" in x))

# Mutant
df["Mutant"] = df["Origin"].str.lower().apply(lambda x: int("mutate" in x or "mutant" in x))

# Normalize origin
def normalize_origin(x):
    x_lower = str(x).lower()
    if "human" in x_lower:
        return "Human"
    elif "mutate" in x_lower or "mutant" in x_lower:
        return "Mutant"
    elif "asgardian" in x_lower:
        return "Asgardian"
    elif "alien" in x_lower:
        return "Alien"
    elif "symbiote" in x_lower:
        return "Symbiote"
    elif "robot" in x_lower:
        return "Robot"
    elif "cosmic being" in x_lower:
        return "Cosmic Being"
    else:
        return "Other"

df["Origin"] = df["Origin"].apply(normalize_origin)

df = df[df["Alive"].isin(["Alive", "Dead"])]
df["Alive"] = (df["Alive"] == "Alive").astype(int)

df = df[num_features + cat_features + [target] + ["PageID"]]

for col in cat_features:
    df[col] = df[col].astype("category")

# Rename PageID to Id for consistency
df = df.rename(columns={"PageID": "Id"})
df["Id"] = df["Id"].astype("str")

# COMMAND ----------
# Split the data
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_set[num_features + cat_features]
y_train = train_set[target]
X_test = test_set[num_features + cat_features]
y_test = test_set[target]

# COMMAND ----------
# Train the model
class CatToIntTransformer(BaseEstimator, TransformerMixin):
    """Transformer that encodes categorical columns as integer codes for LightGBM.

    Unknown categories at transform time are encoded as -1.
    """

    def __init__(self, cat_features: list[str]) -> None:
        """Initialize the transformer with categorical feature names."""
        self.cat_features = cat_features
        self.cat_maps_ = {}

    def fit(self, X: pd.DataFrame, y=None) -> None:
        """Fit the transformer to the DataFrame X."""
        self.fit_transform(X)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform the DataFrame X."""
        X = X.copy()
        for col in self.cat_features:
            c = pd.Categorical(X[col])
            # Build mapping: {category: code}
            self.cat_maps_[col] = dict(zip(c.categories, range(len(c.categories)), strict=False))
            X[col] = X[col].map(lambda val, col=col: self.cat_maps_[col].get(val, -1)).astype("category")
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame X by encoding categorical features as integers."""
        X = X.copy()
        for col in self.cat_features:
            X[col] = X[col].map(lambda val, col=col: self.cat_maps_[col].get(val, -1)).astype("category")
        return X

# COMMAND ----------
preprocessor = ColumnTransformer(
    transformers=[("cat", CatToIntTransformer(cat_features), cat_features)],
    remainder="passthrough"
)

pipeline = Pipeline(
    steps=[("preprocessor", preprocessor),
            ("classifier", LGBMClassifier(**parameters))])

pipeline.fit(X_train, y_train)

# COMMAND ----------
import pickle
import os

os.makedirs("models", exist_ok=True)
# assume pipeline is your trained pipeline
with open("models/model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
