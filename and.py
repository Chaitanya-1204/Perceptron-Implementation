from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd

AND = {
    "X1" : [0 , 0 , 1 , 1] , 
    "X2" : [0 , 1 , 0 , 1] ,
    "Y" :  [0 , 0 , 0 , 1],
    
}

df = pd.DataFrame(AND)

X , Y = prepare_data(df)

ETA = 0.2
epochs = 10


model = Perceptron(eta = ETA , epochs = epochs)

model.fit(X , Y)
