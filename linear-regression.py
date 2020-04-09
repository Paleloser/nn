import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

#  Sources:
#    - .csv - IA NOTEBOOK #1 | Regresión Lineal y Mínimos Cuadrados Ordinarios | Programando IA: https://www.youtube.com/watch?v=w2RJ1D6kz-o&list=PL-Ogd76BhmcCO4VeOlIH93BMT5A_kKAXp
#    - .csv - Regresión Lineal y Mínimos Cuadrados Ordinarios | DotCSV: https://www.youtube.com/watch?v=k964_uNn3l0&list=PL-Ogd76BhmcDxef4liOGXGXLL-4h65bs4&index=6

#                      //////////////
#                      //          //
#  [X_0] --- [W_0] --- //  NEURON  // --- [Y_0]
#              ^       //          //
#              |       //////////////
#          <unknown>         |
#                            |
#                           [B]  <-- <unknown>
#
#
#  Y_0 = W_0 * X_0 + B  => Linear equation
#
#  Y_0 = W_0 * X_0 + B * 1 => Y = [  B, W_0 ] * [   1] = W * X
#                                               [ X_0]

#  Descr: 
#
#  [
#    [CRIM_0, ZN_0, INDUS_0, CHAS_0, NOX_0, RM_0, ..., LSTAT_0],
#    [CRIM_1, ZN_1, INDUS_1, CHAS_1, NOX_1, RM_1, ..., LSTAT_1],
#    [CRIM_2, ZN_2, INDUS_2, CHAS_2, NOX_2, RM_2, ..., LSTAT_2],
#    ...
#    [CRIM_N-1, ZN_N-1, INDUS_N-1, CHAS_N-1, NOX_N-1, RM_N-1, ..., LSTAT_N-1]
#  ]
boston = load_boston()

#  Input matrix: all avg. rooms per home
#
#  (5th index 5] of every row [: )
#
#  [RM_0, RM_1, RM_2, ..., RM_N-1]
#
#  equals
#
#  [X_0, X_1, X_2, ..., X_N-1]
#
#  NOTE: np.array() is just an array casting, as np is an improved library for matrix ops.
X = np.array(boston.data[:, 5])

#  Target matrix: avg. price per home
Y = np.array(boston.target)

plt.scatter(X, Y, alpha=0.5)

#  Add a column full of 1s at the beginning
#
#  Which leaves us X:
#
#     [  1,    X_0]
#     [  1,    X_1]     (X[0]: [  1,   1, ...,     1])
#     [  1,    ...]     (X[1]: [X_0, X_1, ..., X_N-1])
#     [  1,  X_N-1]
#  
#  NOTE: we've got to transpose the X from the dataset + ones to get this one.
X = np.array([np.ones(len(X)), X]).T

#  Now we want to get the least mean squared error (MSE), which is: e = (y_o - y_expect)^2
#
#  NOTE: we use the MSE cause it gives us a positive value that decreases faster when it's smaller 
#  and increases faster when it's bigger.
#
#  Given: y_o = w * x + b => Y = WX (where W = [B, W] AND X = [1, X].T)
# 
#  e = 0 => W = (X.T * X)^-1 * X.T * Y
#
#  NOTE: the * operator for matrixes in .py is '@'
W = np.linalg.inv(X.T @ X) @ X.T @ Y

#  At this point we have found the best values for the weight and bias of our neuron
#  so, if we implemented a neuron with this values and gave it an input (the number of rooms 
#  of a given apt.) it should give us (predict) its belonging output (cost) according to our 
#  model (linear regression).

plt.plot([4, 9], [W[0] + W[1] * 4, W[0] + W[1] * 9], c='black')
plt.show()
