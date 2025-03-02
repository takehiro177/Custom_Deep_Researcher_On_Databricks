# Databricks notebook source
##################################################################################
# Deployment of Command R Model
#
# 
# 
##################################################################################


# COMMAND ----------

# MAGIC %pip install -U --quiet mlflow-skinny mlflow mlflow[gateway]
dbutils.library.restartPython()

# COMMAND ----------

## create mosic ai model serving for cohere chat

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

client.create_endpoint(
    name="cohere-chat-endpoint",
    config={
        "served_entities": [
            {
                "name": "test",
                "external_model": {
                    "name": "command-r-plus-08-2024",
                    "provider": "cohere",
                    "task": "llm/v1/chat",
                    "cohere_config": {
                        "cohere_api_key": "{{secrets/tk-personal/cohere-production}}",
                    }
                }
            }
        ]
    }
)

# COMMAND ----------






# COMMAND ----------




# COMMAND ----------