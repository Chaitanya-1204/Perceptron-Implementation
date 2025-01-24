# Perceptron-Implementation

This repository contains implementation of a Perceptron which is simmply a single nueron with a step function as activation function.

![Perceptron](https://media.geeksforgeeks.org/wp-content/uploads/20230426162726/Perceptron-1.webp)


Forward Pass of Perceptron 

    $z = w_1x_1 + w_2x_2 + w_3x_3 + b$

After the Forward Pass of Perceptron activation Function i.e Step function is implemented

![Step Function](https://miro.medium.com/v2/resize:fit:734/format:webp/0*Q3NIs2MuBgoUMuV0)


    
    
        |---> 1 if z >= 0
    y = |
        |---> 0 otherwise


This output is then comapred with the original truth value and then if there is a error then using the perceptron learning rule weights and bias are updated. 


## Dataset

For the implementation of perceptron we are going to use the Logic gates AND , OR , XOR. 

Truth Table for AND gate 

| A | B | A AND B |
|---|---|---------|
| 0 | 0 |    0    |
| 0 | 1 |    0    |
| 1 | 0 |    0    |
| 1 | 1 |    1    |

Truth Table for OR Gate


| A | B | A OR B |
|---|---|--------|
| 0 | 0 |   0    |
| 0 | 1 |   1    |
| 1 | 0 |   1    |
| 1 | 1 |   1    |



Truth Table for XOR Gate


| A | B | A XOR B |
|---|---|---------|
| 0 | 0 |    0    |
| 0 | 1 |    1    |
| 1 | 0 |    1    |
| 1 | 1 |    0    |



