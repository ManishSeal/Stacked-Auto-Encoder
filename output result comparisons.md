**Comparing results with Standard Classifiers**

The results of classification were compared to the results standard classification algorithm.
On training with one labelled sample per class the prediction accuracy is:
-StackedAuto-Encoder (3 layer): 48.54
-Nearest Neighbour (k=1): 44.98
-Nearest Neighbour (k=3): 21.22
-Nearest Neighbour (k=5): 14.99
-Linear SVM: 44.98
-RBF SVM: 44.98
-Gaussian Process: 36.99
-Decision Tree: 20.37
-Random Forest: 34.98
-Neural Net: 53.91

On training with five labelled sample per class the prediction accuracy is:
-StackedAuto-Encoder (3 layer): 67.32
-Nearest Neighbour (k=1): 64.17
-Nearest Neighbour (k=3): 58.93
-Nearest Neighbour (k=5): 59.45
-Linear SVM: 66.01
-RBF SVM: 38.2
-Gaussian Process: 68.42
-Decision Tree: 44.61
-Random Forest: 50.79
-Neural Net: 69.3

Results were as expected. The ANN of Scikit-Learn outperformed our Stacke Auto-Encoder most probably due to using momnetum while calculating gradients. The stacked auto encoder made here is a pretty simple one. Still such a simple model can outperform the standard classifiers.
