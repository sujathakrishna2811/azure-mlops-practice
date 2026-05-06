
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--sweep_job_name", required=True)

args = parser.parse_args()

sweep_job_name = args.sweep_job_name

job = ml_client.jobs.get(sweep_job_name)


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

