import pandas as pd
import sys 
import yaml
import os 
import numpy as np

from sklearn.preprocessing import FunctionTransformer 
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

## Loading the yaml file 

params = yaml.safe_load(open('params.yaml'))['preprocess'] 

def preprocess(input_path,output_path):
    data = pd.read_csv(input_path)

    X = data.drop(columns='Result')
    y = data['Result']

    # Encode target
    y_encoded = le.fit_transform(y)

    trf = ColumnTransformer(
        [('log transformer',FunctionTransformer(np.log1p),X.columns)],
        remainder='passthrough'
    )

    X_trf = trf.fit_transform(X)

    ## Convert to DataFrame

    X_trf_df = pd.DataFrame(X_trf,columns=X.columns)
    # Add the target variable 'Result' back to the DataFrame
    X_trf_df['Result'] = y_encoded

    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    X_trf_df.to_csv(output_path,index=False)
    print(f"Data preprocess save to {output_path}")

if __name__ == '__main__':
    preprocess(params['input'],params['output'])