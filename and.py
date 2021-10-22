from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np 

AND = {
    "x1" : [0  , 0 ,  1 , 1 ],
    "x2" : [0 , 1 , 0 , 1],
    "Y" : [0 , 0 , 0 , 1]
}
and_df = pd.DataFrame(AND)


X , Y = prepare_data(and_df)

ETA = 0.3
epochs = 10

model = Perceptron(ETA , epochs)
model.fit(X , Y)

loss = model.total_loss()

