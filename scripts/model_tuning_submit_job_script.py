from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.sweep import Choice
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import DefaultAzureCredential

# -----------------------------
# STEP 1: Connect to workspace
# -----------------------------
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="e2486e78-1f52-4aaf-8069-79e7ff451888",
    resource_group_name="AzureAIResourceGroup",
    workspace_name="ml_ws_practice"
)

# -----------------------------
# STEP 2: Get data asset
# -----------------------------
data = ml_client.data.get(
    name="IBM_Telco_Customer_Churn_CSV",
    version="1"
)

# -----------------------------
# STEP 3: Define base command job
# -----------------------------
sweep_job = command(
    code=".",
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
    environment="azureml:telco-sklearn-env@latest",
    compute="mlcomputerclusterpractice",
    display_name="telco-rf-sweep-job",
    experiment_name="telco-rf-sweep"
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

ml_client.jobs.stream(returned_sweep_job.name)

with open("sweep_job_name.txt", "w") as f:
    f.write(returned_sweep_job.name)

print("Sweep job submitted:", returned_sweep_job.name)
print("Status:", returned_sweep_job.status)
print("Studio URL:", returned_sweep_job.studio_url)