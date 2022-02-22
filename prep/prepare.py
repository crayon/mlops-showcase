import argparse
import os
from azureml.core import Run

import numpy as np
from sklearn.model_selection import train_test_split

run = Run.get_context()

parser = argparse.ArgumentParser("prep")

parser.add_argument("--train", type=str, help="train")
parser.add_argument("--test", type=str, help="test")

args = parser.parse_args()

dataframe = run.input_datasets["raw_data"].to_pandas_dataframe()
array = dataframe.values

X = array[:, 0:8]
Y = array[:, 8]

test_size = 0.33
seed = 7

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_size, random_state=seed)

train = np.column_stack((X_train, Y_train))
test = np.column_stack((X_test, Y_test))

os.makedirs(args.train, exist_ok=True)
os.makedirs(args.test, exist_ok=True)

np.savetxt(args.train + "/train.txt", train, fmt="%f")
np.savetxt(args.test + "/test.txt", test, fmt="%f")
