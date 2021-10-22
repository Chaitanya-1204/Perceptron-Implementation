import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from matplotlib.colors import ListedColormap

plt.style.use("fivethirtyeight")

def prepare_data(df):

  '''
    it is used to separate dependent and independent variable 

    Args:
      df(pd.DataFrame) : it is pandas dataFrame

    Return:
      tuple : return a tuple of dependent and independent variable
  
  
  '''
  X = df.drop("Y" , axis  = 1)
  Y = df["Y"]

  return X , Y

def save_model(model , filename):

  '''
      It  saves the model


      Args : 
        model : trained model
        filename : filename of file you want to save
      
      return:
        saved the model

  '''
  model_dir = "models"
  os.makedirs(model_dir ,exist_ok=True)
  filepath = os.path.join(model_dir , filename)
  joblib.dump(model, filepath)

def save_plot(df , filename , model):
  def _create_base_plot(df):
    
    df.plot(kind = "scatter" , x = "x1" , y = "x2" , c = "y" , s = 100 , cmap = "winter")
    
    plt.axhline(y = 0 , color = "black" , linestyle = "--" ,linewidth = 1)
     
    plt.axvline(x = 0 , color = "black" , linestyle = "--" ,linewidth = 1)
    
    figure = plt.gcf() # getting current figure
    
    figure.set_size_inches(10 , 8)
    

  def _plot_decision_regions(X , Y ,classifier , resolution = 0.02):
    
    colors = ("red" , "green" , "blue" , "gray")
    cmap = ListedColormap(colors[:len(np.unique(Y))])

    X = X.values
    x1 = X[:, 0]
    x2 = X[:, 1]
    x1_min , x1_max = x1.min() - 1 , x1.max() + 1
    x2_min , x2_max = x2.min() - 1 , x2.max() + 1
    print(x1_min , x1_max , x2_min , x2_max )
    xx1 , xx2 = np.meshgrid(np.arange(x1_min , x1_max , resolution) ,
                            np.arange(x2_min , x2_max , resolution))
    

    Z =  classifier.predict(np.array([xx1.ravel() , xx2.ravel()]).T)
    
    Z = Z.reshape(xx1.shape)
    
    
    
    plt.contourf(xx1 , xx2 , Z , alpha = 0.2 , cmap = cmap)
    plt.xlim(xx1.min() , xx1.max() )
    plt.ylim(xx2.min() , xx2.max())

    plt.plot()

  X , Y = prepare_data(df)
  _create_base_plot(df)
  _plot_decision_regions(X, Y , model)
  plots_dir = "plots"
  os.makedirs(plots_dir ,exist_ok=True)
  plotPath = os.path.join(plots_dir , filename)
  plt.savefig(plotPath)