#
# Run:
# ```bash
# conda activate deepsatmodels
# python aml_job/submit_training.py
# ```
#

import os
from dotenv import load_dotenv
from azure.ai.ml import command, Input, Output, UserIdentityConfiguration, MLClient
from azure.ai.ml.entities import BuildContext, Data, Environment


# Handle to the workspace
from azure.ai.ml.constants import AssetTypes, InputOutputModes

# Authentication package
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential


ENV_NAME: str = "deepsatmodels_env"
ENV_VERSION: int = 54


try:
    load_dotenv()
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    resource_group_name = os.environ.get("RESOURCE_GROUP")
    workspace_name = os.environ.get("WORKSPACE_NAME")
except Exception as e:
    print(e)
    print(
        "No .env file found. Created one with environment variables SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME"
    )

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    # This will open a browser page
    credential = InteractiveBrowserCredential()


def display_workspace_details(ml_client: MLClient):
    print(f"subscription_id = '{ml_client.subscription_id}'")
    print(f"resource_group_name = '{ml_client.resource_group_name}'")
    print(f"workspace_name = '{ml_client.workspace_name}'")

    print("\nAvailable Environments:")
    for env in ml_client.environments.list():
        print(f" - {env.name}:{env.latest_version}")
    print()

    print("\nLatest Jobs:")
    for index, job in enumerate(ml_client.jobs.list()):
        print(f" - {job.display_name}")
        if index > 5:
            break
    print()


def get_environment(ml_client: MLClient, name: str, version: int) -> Environment:
    print(f"Loading environment `{name}:{version}`...")
    try:
        # https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-pipeline-python-sdk?view=azureml-api-2#create-a-job-environment-for-pipeline-steps
        env = ml_client.environments.get(name, version=f"{version}")
    except:
        print(f"The environment `{name}:{version}` does not already exist. Creating it...")
        env = Environment(
            name=name,
            image="mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04",  # CUDA 10.1 is referenced in the README.md file
            conda_file='deepsatmodels_env.yml',
        )
        env = ml_client.environments.create_or_update(env)
    return env


def get_data_input(ml_client: MLClient, datastore_name: str="covercropatlas", path_on_datastore: str="dataset"):
    datastore_uri: str = f"azureml://subscriptions/{ml_client.subscription_id}/" + \
                         f"resourcegroups/{ml_client.resource_group_name}/" + \
                         f"workspaces/{ml_client.workspace_name}/" + \
                         f"datastores/{datastore_name}/" + \
                         f"paths/{path_on_datastore}/"
    print(f"datastore_uri = '{datastore_uri}'")
    data_folder = Input(
        path=datastore_uri,
        type=AssetTypes.URI_FOLDER,
        mode=InputOutputModes.MOUNT,
    )
    return data_folder


def get_job_args(env, data_folder):
    job_args = {
        "inputs": dict(
            data_folder=data_folder
        ),
        "outputs": dict(),
        "code": "deepsatmodels",  # location of source code
        "command": """python aml_job/train_segmentation_model.py --data_folder ${{inputs.data_folder}}""",
        "environment": env,
        "compute": "FALLOWV100WESTLOW",
        "experiment_name": "deepsatmodels_segmentation_training",
        "display_name": "DeepSatModels Segmentation Training",
        "description": "Training segmentation model using DeepSatModels repository",
        "identity": UserIdentityConfiguration(),
    }

    return job_args


def run():
    # Get a handle to the workspace
    print(f"Using AzureML Workspace `{workspace_name}` within the RG `{resource_group_name}` and Subscription `{subscription_id}`")
    ml_client = MLClient(credential, subscription_id, resource_group_name, workspace_name)

    display_workspace_details(ml_client=ml_client)

    env = get_environment(ml_client=ml_client, name=ENV_NAME, version=ENV_VERSION)
    data_folder = get_data_input(ml_client=ml_client)
    job_args = get_job_args(env=env, data_folder=data_folder)

    job_command = command(**job_args)

    print("Creating job...")
    job = ml_client.create_or_update(job_command)
    print(job)


if __name__ == "__main__":
    run()
    print("DONE")
