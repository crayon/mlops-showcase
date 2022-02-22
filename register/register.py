import argparse
from azureml.core.model import Model
from azureml.core import Run

parser = argparse.ArgumentParser("register")
 
parser.add_argument("--dataset_name", type=str, help="dataset_name")
parser.add_argument("--dataset_version", type=str, help="dataset_version")
parser.add_argument("--metric", type=str, help="metric")
parser.add_argument("--model", type=str, help="model")
parser.add_argument("--model_name", type=str, help="model_name")
 
args = parser.parse_args()
run = Run.get_context()
ws = run.experiment.workspace

avail_model_list = Model.list(
  workspace=ws,
  name=args.model_name,
  tags =[["is_prod", "true"]],
  latest=True
)

prod_model = None
if len(avail_model_list) == 1:
	prod_model = avail_model_list[0]

if (prod_model is not None):
	prod_metric = float(prod_model.tags[args.metric])
	current_metric = float(run.parent.get_metrics().get(args.metric))

	print(
		"Current Production model {}: {}, ".format(args.metric, prod_metric) +
		"New trained model {}: {}".format(args.metric, current_metric)
	)

	if (current_metric >= prod_metric):
		print("New model performs better, registering model...")
		is_prod = "true"
		prod_model.update_tags_properties(
			add_tags={"is_prod": "false"}
		)
	else:
		print("New model performs worse or equal to current model, skipping registration.")
		is_prod = "false"
		run.parent.cancel()
else:
	print("This is first model created, registering...")
	is_prod = "true"


model = Model.register(
  workspace=ws,
  model_path=(args.model + "/model.joblib"),
  model_name=args.model_name,
  tags={
	"dataset_name": args.dataset_name,
	"dataset_version": args.dataset_version,
	"is_prod": is_prod,
	"pipeline_run_id": run.parent.id,
	"pipeline_run_number": run.parent.number,
	args.metric : run.parent.get_metrics().get(args.metric)
  }
)

run.complete()
