from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential
import datetime

# -----------------------------
# STEP 1: Connect to workspace
# -----------------------------
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="xxxxx",
    resource_group_name="xxxx",
    workspace_name="xxxx"
)

# -----------------------------
# STEP 2: Set names
# -----------------------------
endpoint_name = "telco-endpoint" 
deployment_name = "blue"

registered_model_name = "telco-churn-pipeline-model"
registered_model_version = "1"

inference_environment_name = "telco-inference-env"
inference_environment_version = "1"

with open("registered_model_version.txt", "r") as f:
    registered_model_version = f.read().strip()

# -----------------------------
# STEP 3: Create endpoint
# -----------------------------
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Managed online endpoint for telco churn prediction",
    auth_mode="key",
)

ml_client.begin_create_or_update(endpoint).result()
print(f"Endpoint created: {endpoint_name}")

# -----------------------------
# STEP 4: Create deployment
# -----------------------------
deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=endpoint_name,
    model=f"azureml:{registered_model_name}:{registered_model_version}",
    environment=f"azureml:{inference_environment_name}:{inference_environment_version}",
    code_configuration=CodeConfiguration(
        code="src",
        scoring_script="score.py",
    ),
    instance_type="Standard_D2AS_v4",
    instance_count=1,
)

ml_client.begin_create_or_update(deployment).result()
print(f"Deployment created: {deployment_name}")

# -----------------------------
# STEP 5: Route traffic
# -----------------------------
endpoint.traffic = {deployment_name: 100}
ml_client.begin_create_or_update(endpoint).result()

print("Traffic routed successfully.")

# -----------------------------
# STEP 6: Show scoring URI
# -----------------------------
online_endpoint = ml_client.online_endpoints.get(endpoint_name)
print("Endpoint name:", online_endpoint.name)
print("Scoring URI:", online_endpoint.scoring_uri)
