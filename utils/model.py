import numpy as np
import logging

class Perceptron:
  def __init__(self , eta , epochs): # eta = learning_rate 
    self.weights = np.random.randn(3) * 1e-4 # Small weight initialization
    logging.info(f"Inital weights before training : {self.weights} " )
    self.eta =eta # Learning Rate
    self.epochs = epochs 


  def activationFunction(self , inputs , weights):
    z = np.dot(inputs , weights)  
    
    return np.where(z > 0 , 1, 0)
  
  def fit(self ,X, Y):
   
    self.X = X
    
    self.Y = Y
    
    X_with_bias = np.c_[X , -np.ones((len(X) , 1))]
   
    logging.info(f"X with bias : \n{X_with_bias}")
    
    for epoch in range(self.epochs):
      logging.info("\n\n")
      logging.info("--"*10)
      logging.info(f"For epoch : {epoch}")
      logging.info("--" *10)

      y_hat = self.activationFunction(X_with_bias , self.weights)  # forward Propogation 
      logging.info(f"Predicted value after forward Pass : {y_hat}")
      
      self.error = self.Y - y_hat
      logging.info(f"error : \n{self.error} ")

       
      self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error) # backward Propogation 
     
      logging.info(f"updated weights after epoch  {epoch}  : {self.weights}")
      

  def predict(self , X):
    X_with_bias = np.c_[X , -np.ones((len(X) , 1 ) )]
    return self.activationFunction(X_with_bias , self.weights)

  def total_loss(self):
    total_loss = np.sum(self.error)
    logging.info(f"total loss : {total_loss}")
    return total_loss