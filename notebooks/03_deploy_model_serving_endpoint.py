# Databricks notebook source

import os

import mlflow
import numpy as np
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    EndpointTag,
    ServedEntityInput,
)
from dotenv import load_dotenv
from mlflow import MlflowClient
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig

# COMMAND ----------

w = WorkspaceClient()
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

model_name = f"{config.catalog_name}.{config.schema_name}.marvel_character_model_basic"

client = MlflowClient()
model_version = client.get_model_version_by_alias(alias="latest-model", name=model_name)

served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        scale_to_zero_enabled=True,
        workload_size="Small",
        entity_version=model_version.version,
    )
]

workspace = WorkspaceClient()
endpoint_name = "marvel-character-model-serving"
endpoint_exists = any(item.name ==endpoint_name for item in workspace.serving_endpoints.list())


if not endpoint_exists:
    workspace.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=served_entities,
        ),
        tags=[EndpointTag.from_dict({"key": "project_name", "value": "marvel_characters"})]
    )
else:
    workspace.serving_endpoints.update_config(name=endpoint_name, served_entities=served_entities)

# COMMAND ----------
# Get sample records
spark = SparkSession.builder.getOrCreate()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Sample records from the training set
required_columns = [
    "Height",
    "Weight",
    "Universe",
    "Identity",
    "Gender",
    "Marital_Status",
    "Teams",
    "Origin",
    "Magic",
    "Mutant"]
sampled_records = test_set[required_columns].sample(n=18000, replace=True)

# Replace NaN values with None (which will be serialized as null in JSON)
sampled_records = sampled_records.replace({np.nan: None}).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------
# Call endpoint
"""
Each dataframe record in the request body should be list of json with columns looking like:

[{'Height': 1.75,
  'Weight': 70.0,
  'Universe': 'Earth-616',
  'Identity': 'Public',
  'Gender': 'Male',
  'Marital_Status': 'Single',
  'Teams': 'Avengers',
  'Origin': 'Human',
  'Creators': 'Stan Lee'}]
"""

serving_endpoint = f"{host}/serving-endpoints/marvel-character-model-serving/invocations"

response = requests.post(
    serving_endpoint,
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)


print(f"Response Status: {response.status_code}")
print(f"Response Text: {response.text}")


# COMMAND ----------
