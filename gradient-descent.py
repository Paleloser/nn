import numpy as np
import matplotlib.pyplot as plt

# Defines a plane with multiple minimums
func = lambda th: np.sin((th[0] ** 2) / 2 - (th[1] ** 2) / 4 + 3) * np.cos(2 * th[0] + 1 - np.exp(th[1]))

# Plot the plane
res = 100
bound = 2
th = [np.linspace(-bound, bound, res), np.linspace(-bound, bound, res)]
_Z = np.zeros((res, res))

for i, x in enumerate(th[0]):
    for j, y in enumerate(th[0]):
        _Z[j, i] = func([x, y])

plt.contourf(th[0], th[1], _Z, res)
plt.colorbar()

# Random starting x and y coordinates T = [x, y]
T = np.random.random(2) * 4 - 2
# Delta to calculate the derivative of a function in a point
delta = 0.001
# Gradient vector grad(f) = [df/dx, df/dy]
grad = np.zeros(2)
# Hou much does the gradient affects == step size moving to the minimum
# Note that if this step is too big we may jump over the minimum, and if
# it is too small, we may not get to the minimum in out irerations.
lr = 0.05

plt.plot(T[0], T[1], '.', c='white')

# Epochs
for _ in range(10000):
    # The domain of our function is in 2 dimensions: x and y => 2 part. derivs.
    for i, th in enumerate(T):
        # Partial Deriv of a function in a given point:
        #       f(x + h) - f(x)
        #   f' = --------------
        #              h
        # Note that we only increase the function in the desired component (the one
        # tht we are deriving)
        _T = np.copy(T)
        _T[i] = _T[i] + delta
        deriv = (func(_T) - func(T)) / delta
        # The component of the gradient vector
        grad[i] = deriv
    # If a component of a gradient vector is negative it will mean that that a minimum
    # is on that direction. Therefore qe want to keep going that way.
    T = T - lr * grad

    if (_ % 100 == 0):
        plt.plot(T[0], T[1], '.', c='red')

plt.plot(T[0], T[1], '.', c='black')
plt.show()
