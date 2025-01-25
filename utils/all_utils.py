
import pandas as pd 
import os 
import joblib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def prepare_data(df):
    
    X = df.drop("Y" , axis = 1)
    Y = df["Y"]
    return X , Y

def save_model(model , filename):
    model_dir = "models"
    os.makedirs(model_dir ,exist_ok = True)
    filepath = os.path.join(model_dir , filename)
    joblib.dump(model , filepath)
    

def save_plot(df , file_name , model):
    
    def _create_base_plot(df):
       
        df.plot(kind = "scatter" , x = "X1" , y = "X2" , c = "Y" , s = 100 , cmap = "winter")
        plt.axhline(y = 0 , color = "black" , linestyle = "--" , linewidth = 1)
        plt.axvline(x = 0 , color = "black" , linestyle = "--" , linewidth = 1)
        figure = plt.gcf()
        figure.set_size_inches(10 , 8)
        
    
    def _plot_decision_regions(X , Y , classifier , resolution = 0.02):
        colors = ["red" , "blue" , "lightgreen" , "gray" , "cyan"]
        
        cmap = ListedColormap(colors[:len(np.unique(Y))])
        X = X.values
        x1 = X[: , 0]
        x2 = X[: , 1]
        x1_min , x1_max = x1.min() - 1 , x1.max() + 1
        x2_min , x2_max = x2.min() - 1 , x2.max() + 1
        xx1 , xx2 = np.meshgrid(np.arange(x1_min , x1_max  , resolution) , 
                                np.arange(x2_min , x2_max , resolution))
        
        print(xx1)
        print("---" * 10)
        print(xx1.ravel())
        
        Z = classifier.predict(np.array([xx1.ravel() , xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contour(xx1 , xx2 , Z , alpha = 0.2 , cmap = cmap)
        plt.xlim(xx1.min() , xx1.max())
        plt.ylim(xx2.min() , xx2.max())
        plt.plot()
        
    X , Y = prepare_data(df)
    
    _create_base_plot(df)
    _plot_decision_regions(X , Y , model)
    
    plot_dir = "plots"
    os.makedirs(plot_dir , exist_ok = True)
    plotPath = os.path.join(plot_dir , file_name)
    
    plt.savefig(plotPath)
        
    