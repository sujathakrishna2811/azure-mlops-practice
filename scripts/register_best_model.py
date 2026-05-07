
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os
import time

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_ML_WORKSPACE"]
)

parser = argparse.ArgumentParser()
parser.add_argument("--sweep_job_name", required=True)
args = parser.parse_args()
sweep_job_name = args.sweep_job_name
while True:
    job = ml_client.jobs.get(sweep_job_name)
    print("Sweep job status:", job.status)

    if job.status in ["Completed", "Failed", "Canceled"]:
        break

    time.sleep(60)

if job.status != "Completed":
    raise Exception(f"Sweep job did not complete successfully. Status: {job.status}")

print("Job properties:", job.properties)

best_job_name = job.properties["best_child_run_id"]
print("Best job:", best_job_name)

#Register BEST model (automatic)
model = Model(
    path=f"azureml://jobs/{best_job_name}/outputs/artifacts/paths/outputs/model_pipeline.pkl",
    name="telco-churn-pipeline-model",
    type=AssetTypes.CUSTOM_MODEL,
    description="Best Random Forest pipeline model from sweep job",
    tags={         
        "project": "telco-churn",
        "selection_metric": "val_auc"
        }
)
registered_model = ml_client.models.create_or_update(model)

print("Best child job:", best_job_name)
print("Registered Model:", registered_model.name)
print("Version:", registered_model.version)

