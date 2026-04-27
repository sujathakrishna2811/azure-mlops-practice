from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes


ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="e2486e78-1f52-4aaf-8069-79e7ff451888",
    resource_group_name="AzureAIResourceGroup",
    workspace_name="ml_ws_practice"
)

with open("sweep_job_name.txt", "r") as f:
    sweep_job_name = f.read().strip()

returned_sweep_job = ml_client.jobs.get(sweep_job_name)
best_job_name = returned_sweep_job.properties["best_child_run_id"]

model = Model(
    path=f"azureml://jobs/{best_job_name}/outputs/artifacts/paths/outputs/model_pipeline.pkl",
    name="telco-churn-pipeline-model",
    type=AssetTypes.CUSTOM_MODEL,
    description="Best Random Forest pipeline model from sweep job",
    tags={
        "project": "telco-churn",
        "selection_metric": "accuracy"
    }
)

registered_model = ml_client.models.create_or_update(model)

with open("registered_model_version.txt", "w") as f:
    f.write(str(registered_model.version))

print("Registered Model:", registered_model.name)
print("Version:", registered_model.version)
