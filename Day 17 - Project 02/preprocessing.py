import numpy as np
import pandas as pd

import argparse
import os

import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action="ignore", category=DataConversionWarning)

columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
       'Viscera weight', 'Shell weight', 'Rings']

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.2)
    args, _ = parser.parse_known_args()
    print(f"recieved args: {args}")
    
    input_data_path = os.path.join("/opt/ml/processing/input", "abalone.csv")
    
    data = pd.read_csv(input_data_path)
    data = pd.DataFrame(data, columns=columns)
    
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    data["Sex"] = encoder.fit_transform(data["Sex"])
    scaler = StandardScaler()
    
    
    x = data.drop(["Rings"], axis=1)
    y = data.Rings
    
    continious = x.columns[1:]
    def outlier_handler(col):
        d = x[col]
        q1 = d.quantile(0.25)
        q3 = d.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        d = d.clip(lower, upper)
    
    for c in continious:
        outlier_handler(c)
    
    for c in continious:
        x[[c]] = scaler.fit_transform(x[[c]])
    


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