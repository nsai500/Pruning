# Pruning
Given a layer of a neural network ReLU(xW) are two well-known ways to prune it:
- Weight pruning: set individual weights in the weight matrix to zero. Here, to achieve sparsity of k% we rank the individual weights in weight matrix W according to their magnitude (absolute value) |w<sub>i,j</sub>|, and then set to zero the smallest k%.
- Unit/Neuron pruning: set entire columns to zero in the weight matrix to zero, in
effect deleting the corresponding output neuron. Here to achieve sparsity of k%, we rank the the columns of a weight
matrix according to their L2-norm |w| = ![](http://latex.codecogs.com/gif.latex?%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%28x_%7Bi%7D%29%5E%7B2%7D%7D) and delete the smallest k%.

# Results
Weight Pruning

| k%            | Accuracy%     |
| ------------- | ------------- |
| 0             | 98.08         |
| 25            | 98.02         |
| 50            | 98.05         |
| 60            | 98.01         |
| 70            | 97.83         |
| 80            | 97.59         |
| 90            | 95.49         |
| 95            | 85.32         |
| 97            | 57.90         |
| 99            | 17.06         |

![](https://github.com/nsai500/Pruning/blob/master/weight_pruning.png?raw=true)

Unit Pruning

| k%            | Accuracy%     |
| ------------- | ------------- |
| 0             | 98.08         |
| 25            | 98.08         |
| 50            | 98.01         |
| 60            | 97.56         |
| 70            | 95.65         |
| 80            | 83.03         |
| 90            | 38.02         |
| 95            | 20.05         |
| 97            | 15.47         |
| 99            | 10.03         |

![](https://github.com/nsai500/Pruning/blob/master/unit_pruning.png?raw=true)

# Observations
From the graphs, we can observe that the weight pruning method works better in identifying the weights which don't contribute to the output than the unit pruning method. This might be due to the fact that unit pruning removes entire columns which have lesser L2-norm even if they contain few important weights.

In the Weight pruning method, even after pruning 90% of weights in each layer, the accuracy remains at 95.49%. We can say only 10%(approx. 238,600 out of 2,386,000 parameters) are contributing to the final output. This sparsity can be used to decrease network size and to increase performance.

In the Unit pruning method, after dropping 70% of weights in each layer, the accuracy remains at 95.65%. This suggests that there are many neurons in the network that don't contribute to the output.

After using the same technique on networks with lesser number of layers, I observed that the % of weights that can be dropped without hurting performance is higher for networks with more number of layers.
# Guide
pruning.py file contains all the code for the pruning.

Currently the file uses the pre_trained weights(pre_trained.h5), comment the load weights( line 51) and uncomment line48,49 to retrain the model for any modifications.
