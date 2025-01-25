import numpy as np


class Perceptron:
    
    def __init__(self , epochs , eta):
        self.weights = np.random.randn(3) * 1e-4 # Small weight Initialization
       
        self.epochs = epochs 
        self.eta = eta # Learninig Rate
    
    def activation_function(self , inputs , weights):
        z = np.dot(inputs , weights)
        return np.where(z > 0 , 1 , 0)
    
    def fit(self , X , Y):
        self.X  = X 
        self.Y = Y
        
        # Concatenating a column of -1 for the bias term
        X_with_bias = np.c_[self.X , -np.ones((len(self.X) , 1) )] 
        print(f"X_with_bias : \n{X_with_bias}")
        
        for epoch in range(self.epochs):
            print("--" * 10)
            print(f"for Epoch : {epoch}")
            print("--" * 10)
            
            # Forward Pass
            y_hat = self.activation_function(X_with_bias , self.weights)
            print(f"Predicted value after forward pass : \n{y_hat}")
            
            # Calculating Error
            
            self.error = self.Y - y_hat
            print(f"The error is : \n{self.error}")
            
            # Backpropogation
            
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T , self.error)
            
            print(f"Updated Weights after epochs : {epoch}/{self.epochs} \n{self.weights}")
            
            print("####" * 10)
            
    
    def predict(self, X):
        
        X_with_bias = np.c_[X , -np.ones((len(X) , 1))]
        return self.activation_function(X_with_bias , self.weights)
    
    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"total_loss : {total_loss}")
        return total_loss
    
            
            
