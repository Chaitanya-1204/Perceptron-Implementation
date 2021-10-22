from utils.model import Perceptron
from utils.all_utils import prepare_data , save_model , save_plot 
import pandas as pd
import numpy as np 

AND = {
    "x1" : [0  , 0 ,  1 , 1 ],
    "x2" : [0 , 1 , 0 , 1],
    "Y" : [0 , 0 , 0 , 1]
}
and_df = pd.DataFrame(AND)
print(and_df)

X , Y = prepare_data(and_df)

ETA = 0.3
epochs = 10

model = Perceptron(ETA , epochs)
model.fit(X , Y)

loss = model.total_loss()

save_model(model , "and.model")
save_plot(and_df , "and.jpg" , model)
