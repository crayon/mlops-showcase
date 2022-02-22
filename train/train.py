import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os
from azureml.core import Run

parser = argparse.ArgumentParser("train")

parser.add_argument("--train", type=str, help="train")
parser.add_argument("--test", type=str, help="test")
parser.add_argument("--metric", type=str, help="metric")
parser.add_argument("--model", type=str, help="model")

args = parser.parse_args()
run = Run.get_context()


train = np.loadtxt(args.train + "/train.txt", dtype=float)
test = np.loadtxt(args.test + "/test.txt", dtype=float)

X_train = train[:, 0:8]
Y_train = train[:, 8]

X_test = test[:, 0:8]
Y_test = test[:, 8]

model = LogisticRegression(max_iter=100000)
model.fit(X_train, Y_train)
if not os.path.isdir(args.model):
    os.mkdir(args.model)
joblib.dump(model, args.model + "/model.joblib")
result = model.score(X_test, Y_test)

run.log(args.metric, result)
run.parent.log(args.metric, result)

run.complete()
