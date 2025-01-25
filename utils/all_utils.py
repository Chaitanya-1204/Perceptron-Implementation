
import pandas as pd 


def prepare_data(df):
    
    X = df.drop("Y" , axis = 1)
    Y = df["Y"]
    return X , Y