# MLOps workshop at SRECon 8 Oct 2025

Use Databricks free edition: https://docs.databricks.com/aws/en/getting-started/free-edition

# Install
Databricks CLI: https://docs.databricks.com/aws/en/dev-tools/cli/install
UV: https://docs.astral.sh/uv/getting-started/installation/

# Create Databricks profile
databricks auth login --host <YOUR_HOST>

# Create env
uv sync --extra dev

# Data
Using the [**Marvel Characters Dataset**](https://www.kaggle.com/datasets/mohitbansal31s/marvel-characters?resource=download) from Kaggle.

This dataset contains detailed information about Marvel characters (e.g., name, powers, physical attributes, alignment, etc.).
It is used to build classification and feature engineering models for various MLOps tasks, such as predicting character attributes or status.
