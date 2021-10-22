from utils.model import Perceptron
from utils.all_utils import prepare_data , save_model , save_plot 
import pandas as pd
import numpy as np 

OR = {
    "x1" : [0  , 0 ,  1 , 1 ],
    "x2" : [0 , 1 , 0 , 1],
    "Y" : [0 , 1 , 1 , 1]
}
or_df = pd.DataFrame(OR)
print(or_df)

X , Y = prepare_data(or_df)

ETA = 0.3
epochs = 10

model = Perceptron(ETA , epochs)
model.fit(X , Y)

loss = model.total_loss()

save_model(model , "or.model")
save_plot(or_df , "or.jpg" , model)
