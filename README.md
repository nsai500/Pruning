# Pruning
Given a layer of a neural network ReLU(xW) are two well-known ways to prune it:
- Weight pruning: set individual weights in the weight matrix to zero. Here, to achieve sparsity of k% we rank the individual weights in weight matrix W according to their magnitude (absolute value) |w<sub>i,j</sub>|, and then set to zero the smallest k%.
- Unit/Neuron pruning: set entire columns to zero in the weight matrix to zero, in
effect deleting the corresponding output neuron. Here to achieve sparsity of k%, we rank the the columns of a weight
matrix according to their L2-norm |w| = ![](http://latex.codecogs.com/gif.latex?%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%28x_%7Bi%7D%29%5E%7B2%7D%7D) and delete the smallest k%.

# Results
![](https://github.com/nsai500/Pruning/blob/master/weight_pruning.png?raw=true)

![](https://github.com/nsai500/Pruning/blob/master/unit_pruning.png?raw=true)

# Observations
