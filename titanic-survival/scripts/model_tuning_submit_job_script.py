from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.sweep import Choice
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import DefaultAzureCredential
import os

# -----------------------------
# STEP 1: Connect to workspace
# -----------------------------
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_ML_WORKSPACE"]
)

# -----------------------------
# STEP 2: Get data asset
# -----------------------------
data = ml_client.data.get(
    name="Titanic_Survival_Dataset",
    version="1"
)

# -----------------------------
# STEP 3: Define base command job
# -----------------------------
sweep_job = command(
    code="titanic-survival",
    command=(
        "python src/train_tuning.py "
        "--data ${{inputs.data}} "
        "--n_estimators ${{inputs.n_estimators}} "
        "--max_depth ${{inputs.max_depth}} "
        "--min_samples_split ${{inputs.min_samples_split}}"
    ),
    inputs={
        "data": Input(
            type=AssetTypes.URI_FILE,
            path=data.path,
            mode=InputOutputModes.RO_MOUNT
        ),
        "n_estimators": Choice(values=[50, 100, 200]),
        "max_depth": Choice(values=[5, 10, 20]),
        "min_samples_split": Choice(values=[2, 5, 10]),
    },
    environment="azureml:titanic-sklearn-env:1",
    compute="mlcomputerclusterpractice",
    display_name="titanic-rf-sweep-job",
    experiment_name="titanic-rf-sweep"
).sweep(
    sampling_algorithm="random",
    primary_metric="accuracy",
    goal="Maximize"
)

# -----------------------------
# STEP 4: Set sweep limits
# -----------------------------
sweep_job.set_limits(
    max_total_trials=12,
    max_concurrent_trials=3,
    timeout=3600
)

# -----------------------------
# STEP 5: Submit sweep job
# -----------------------------
returned_sweep_job = ml_client.jobs.create_or_update(sweep_job)

print("Sweep job submitted:", returned_sweep_job.name)
print("Status:", returned_sweep_job.status)
print("Studio URL:", returned_sweep_job.studio_url)

with open("sweep_job_name.txt", "w") as f:
    f.write(returned_sweep_job.name)
