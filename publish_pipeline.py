import os
from azureml.core import Dataset, Datastore, Environment, RunConfiguration, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import AmlCompute
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PipelineData, Pipeline


auth_resource_group = os.environ.get('AUTH_RESOURCE_GROUP')
auth_sp_id = os.environ.get('AUTH_SP_ID')
auth_sp_secret = os.environ.get('AUTH_SP_SECRET')
auth_subscr_id = os.environ.get('AUTH_SUBSCR_ID')
auth_tenant_id = os.environ.get('AUTH_TENANT_ID')
auth_workspace_name = os.environ.get('AUTH_WORKSPACE_NAME')
build_id = os.environ.get('GITHUB_RUN_ID')
compute_name = "waicf-cluster"
datastore_name = "waicfblobstorage"
dataset_name = "waicf-diabetes"
env_name = "waicf-diabetes"
env_inference_name = "waicf-diabetes-infer"
inference_name = "waicf"
metric = "score"
model_name = "waicf-diabetes"
pipeline_name = "waicf-pipeline"


credentials = ServicePrincipalAuthentication(
  tenant_id=auth_tenant_id,
  service_principal_id=auth_sp_id,
  service_principal_password=auth_sp_secret
)
ws = Workspace(
  subscription_id=auth_subscr_id,
  resource_group=auth_resource_group,
  workspace_name=auth_workspace_name,
  auth=credentials
)
env = Environment.from_conda_specification(env_name, "./conda.yml")
compute_cluster = AmlCompute(
    workspace=ws,
    name=compute_name,
)

run_config = RunConfiguration()
run_config.target = compute_cluster
run_config.environment = env

datastore = Datastore.get(
    workspace=ws,
    datastore_name=datastore_name,
)

diabetes_data = Dataset.Tabular.from_delimited_files(
    path=[datastore.path("./diabetes1.csv"), datastore.path("./diabetes2.csv")])
diabetes_dataset = diabetes_data.register(
    workspace=ws,
    name=dataset_name,
    create_new_version=True,
)

raw_data = diabetes_data.as_named_input("raw_data")
train_data = PipelineData("train_data", datastore=datastore).as_dataset()
test_data = PipelineData("test_data", datastore=datastore).as_dataset()
model_file = PipelineData("model_file", datastore=datastore)
register_deploy_dep = PipelineData("dependency", datastore=datastore)


step1 = PythonScriptStep(
    name="Data preparation",
    source_directory="./prep",
    script_name="prepare.py",
    arguments=[
        "--test", test_data,
        "--train", train_data,
    ],
    compute_target=compute_cluster,
    runconfig=run_config,
    allow_reuse=False,
    inputs=[raw_data],
    outputs=[
        train_data,
        test_data,
    ],
)

step2 = PythonScriptStep(
    name="Train the model",
    source_directory="./train",
    script_name="train.py",
    arguments=[
        "--metric", metric,
        "--model", model_file,
        "--train", train_data,
        "--test", test_data,
    ],
    compute_target=compute_cluster,
    runconfig=run_config,
    allow_reuse=False,
    inputs=[
        train_data,
        test_data,
    ],
    outputs=[model_file],
)

step3 = PythonScriptStep(
    name="Register the model",
    source_directory="./register",
    script_name="register.py",
    arguments=[
        "--dataset_name", diabetes_dataset.name,
        "--dataset_version", diabetes_dataset.version,
        "--metric", metric,
        "--model", model_file,
        "--model_name", model_name,
    ],
    inputs=[model_file],
    outputs=[register_deploy_dep],
    compute_target=compute_cluster,
    runconfig=run_config,
    allow_reuse=True,
)

step4 = PythonScriptStep(
    name="Deploy the model",
    source_directory="./deploy",
    script_name="deploy.py",
    arguments=[
        "--env_name", env_inference_name,
        "--inference_name", inference_name,
        "--model_name", model_name,
    ],
    inputs=[register_deploy_dep],
    compute_target=compute_cluster,
    runconfig=run_config,
    allow_reuse=True,
)

pipeline_steps = [step1, step2, step3, step4]

pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
published_pipeline = pipeline.publish(name=pipeline_name, version=build_id)

print(f"Published pipeline: {published_pipeline.name}")
print(f"for build {published_pipeline.version}")
