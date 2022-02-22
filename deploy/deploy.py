import argparse
from azureml.core.model import InferenceConfig, Model
from azureml.core import Run, Environment
from azureml.core.webservice import AciWebservice, AksWebservice

parser = argparse.ArgumentParser("deploy")

parser.add_argument("--env_name", type=str, help="env_name")
parser.add_argument("--inference_name", type=str, help="inference_name")
parser.add_argument("--model_name", type=str, help="model_name")

args = parser.parse_args()
run = Run.get_context()
ws = run.experiment.workspace

model = Model(
  workspace=ws,
  name=args.model_name,
  tags = [["is_prod", "true"]],
)

env = Environment.from_conda_specification(
  name=args.env_name,
  file_path="./conda.yml",
)

inference_config = InferenceConfig(
  source_directory="./",
  entry_script="score.py",
  environment=env,
)

deployment_config = AciWebservice.deploy_configuration(
  cpu_cores=1,
  memory_gb=1,
)

endpoint = Model.deploy(
  workspace=ws,
  name=args.inference_name,
  models=[model],
  inference_config=inference_config,
  deployment_config=deployment_config,
  overwrite=True,
)

endpoint.wait_for_deployment(show_output=True)

run.complete()
