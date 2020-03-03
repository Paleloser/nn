import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

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
#  Y_0 = W_0 * X_0 + B * 1 => Y = [  1, X_0 ] * [   B] = X * W
#                                               [ W_0]

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

#  Now we want to get the least mean squared error, which is: e = (y_o - y_expect)^2
#
#  (y_o - y_expect)^2 = y_o^2 - 2 * y_o * y_expect + y_expect^2
# 
#  Given: y_o = w * x + b => Y = WX (where W = [B, W] AND X = [1, X])
# 
#  e = (WX)^2 - 2WX + Y^2 -> e = 0 => W = (X^T * X)^-1 * X^T * Y
B = np.linalg.inv(X.T @ X) @ X.T @ Y

plt.plot([4, 9], [B[0] + B[1] * 4, B[0] + B[1] * 9], c='black')
plt.show()
