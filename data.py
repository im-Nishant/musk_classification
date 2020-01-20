import os
import random

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from joblib import dump, load


def prepare_data_from_csv(
    data_file_path: str,
    drop_columns: list,
    output_column: str='class',
    scale: bool=True,
    scale_range: tuple=(0,1),
    shuffle: bool=True,
    test_split_ratio: float=0.2,
    one_hot: bool=False
    ):
    try:
        data = pd.read_csv(data_file_path)
        data.drop(drop_columns, inplace=True)
        # if shuffle is True:
        #     data = data.sample(frac=1).reset_index(drop=True)
        
        y_data = data.pop(output_column)
        x_data = data

        if scale:
            scaler = preprocessing.MinMaxScaler(feature_range=scale_range)
            x_data = scaler.fit_transform(x_data)
            dump(scaler, 'scaling_data_func.joblib')
        
        if one_hot:
            y_onehot_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
            y_onehot = y_onehot_encoder.fit_transform(
                np.array(y_data).reshape((-1,1))
            ).toarray()
            dump(y_onehot_encoder, 'one_hot_output_encoder.joblib')
            # x_train, x_test, y_train, y_test, y_onehot_train, y_onehot_test
            return(
                train_test_split(
                    x_data, 
                    y_data,
                    y_onehot, 
                    test_size=test_split_ratio, 
                    shuffle=shuffle
                )
            )
        
        # x_train, x_test, y_train, y_test
        return(
            train_test_split(
                x_data,
                y_data,
                test_size=test_split_ratio,
                shuffle=shuffle
            )
        )
    except:
        print("Error")
        

        
        
            
