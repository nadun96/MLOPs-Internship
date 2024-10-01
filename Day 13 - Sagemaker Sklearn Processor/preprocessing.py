import numpy as np
import pandas as pd

import argparse
import os

import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action="ignore", category=DataConversionWarning)

columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.2)
    args, _ = parser.parse_known_args()
    print(f"recieved args: {args}")
    
    input_data_path = os.path.join("/opt/ml/processing/input", "heart.csv")
    
    data = pd.read_csv(input_data_path)
    data = pd.DataFrame(data, columns=columns)
    
    mmscaler = MinMaxScaler()
    data[["chol", "oldpeak", "thalach"]] = mmscaler.fit_transform(data[["chol", "oldpeak", "thalach"]])
    sscaler = StandardScaler()
    data[["age", "trestbps"]] = sscaler.fit_transform(data[["age", "trestbps"]])
    x = data.drop(["target"], axis=1)
    y = data.target
    split_ratio = args.train_test_split_ratio
    print(f"split: {split_ratio}")
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=split_ratio, random_state=42)
    
    
    Xtrain_output_path = os.path.join("/opt/ml/processing/train", "Xtrain.csv")
    Xtest_output_path = os.path.join("/opt/ml/processing/train", "Xtest.csv")
    ytrain_output_path = os.path.join("/opt/ml/processing/test", "ytrain.csv")
    ytest_output_path = os.path.join("/opt/ml/processing/test", "ytest.csv")
   
    print("Saving features:")
    
    pd.DataFrame(x_train).to_csv(Xtrain_output_path, header=False, index=False)
    pd.DataFrame(x_test).to_csv(Xtest_output_path, header=False, index=False)
    pd.DataFrame(y_train).to_csv(ytrain_output_path, header=False, index=False)
    pd.DataFrame(y_test).to_csv(ytest_output_path, header=False, index=False)