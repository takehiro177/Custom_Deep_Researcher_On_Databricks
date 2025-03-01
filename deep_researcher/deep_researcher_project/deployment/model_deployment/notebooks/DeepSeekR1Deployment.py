# Databricks notebook source
##################################################################################
# Deployment of DeepSeek R1 Model
#
# 
# 
##################################################################################



# COMMAND ----------
# MAGIC %pip install transformers==4.44.2 mlflow==2.20.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("model_id", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "Name of Huggingface Model")

model_id = dbutils.widgets.get("model_id")


# COMMAND ----------
import os

LOCAL_DISK_HF = "/local_disk0/hf_cache"
os.makedirs(LOCAL_DISK_HF, exist_ok=True)
os.environ["HF_HOME"] = LOCAL_DISK_HF
os.environ["HF_DATASETS_CACHE"] = LOCAL_DISK_HF
os.environ["TRANSFORMERS_CACHE"] = LOCAL_DISK_HF


# COMMAND ----------

from huggingface_hub import snapshot_download
snapshot_download(model_id)


# COMMAND ----------


import mlflow
import transformers

mlflow.set_registry_uri("databricks-uc")

my_uc_catalog = "dev"
my_uc_schema = "deep_researcher"
uc_model_name = "deepseek_r1_distilled_llama8b_v1"

task = "llm/v1/chat"
model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

transformers_model = {"model": model, "tokenizer": tokenizer}

with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=transformers_model,
        artifact_path="model",
        task=task,
        registered_model_name=f"{my_uc_catalog}.{my_uc_schema}.{uc_model_name}",
        metadata={
        "pretrained_model_name": "meta-llama/Llama-3.1-8B-Instruct",
           "databricks_model_family": "LlamaForCausalLM",
           "databricks_model_size_parameters": "8b",
       },
    )


