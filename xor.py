from utils.model import Perceptron
from utils.all_utils import prepare_data , save_model , save_plot 
import pandas as pd
import numpy as np 

XOR = {
    "x1" : [0  , 0 ,  1 , 1 ],
    "x2" : [0 , 1 , 0 , 1],
    "Y" : [0 , 1 , 1 , 0]
}
xor_df = pd.DataFrame(XOR)
print(xor_df)

X , Y = prepare_data(xor_df)

ETA = 0.3
epochs = 10

model = Perceptron(ETA , epochs)
model.fit(X , Y)

loss = model.total_loss()

save_model(model , "xor.model")
save_plot(xor_df , "xor.jpg" , model)
