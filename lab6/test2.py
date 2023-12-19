import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg

def ESN(data, first_steps, train_length, test_length, nr_reservoir, spectral_radius, reg_coefficient, seed=42):
    np.random.seed(seed)
    Nu = 1 
    Win = np.random.uniform(-1, 1, (nr_reservoir, Nu + 1)) * 0.2
    W = np.random.uniform(-1, 1, (nr_reservoir, nr_reservoir))
    rhoW = max(abs(linalg.eig(W)[0]))
    W *= spectral_radius / rhoW

    X = np.zeros((1 + Nu + nr_reservoir, train_length - first_steps))
    Yt = data[None, first_steps+1:train_length+1]

    x = np.zeros((nr_reservoir, 1))
    for t in range(train_length):
        u = data[t]
        x = np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
        if t >= first_steps:
            X[:, t - first_steps] = np.vstack((1, u, x))[:, 0]

    Wout = linalg.solve(np.dot(X, X.T) + reg_coefficient * np.eye(1 + Nu + nr_reservoir), np.dot(X, Yt.T)).T

    # Run the trained ESN in a generative mode
    Y = np.zeros((1, test_length))
    u = data[train_length]
    for t in range(test_length):
        x = np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
        y = np.dot(Wout, np.vstack((1, u, x)))
        Y[:, t] = y
        u = y

    # Compute MSE for the test time steps
    mse = sum(np.square(data[train_length:train_length+test_length] - Y[0, :])) / test_length

    return Wout, Y, mse

train_length = 3000
test_length = 1000
nr_reservoir = 1000
spectral_radius = 0.8
reg_coefficient = 1e-8

# Generate signal
n = np.arange(1, 4001)
data = 0.5 * np.sin(n / 4)

# Run ESN model
Wout, Y, mse = ESN(
    data, 
    first_steps=1000, 
    train_length=train_length,
    test_length=test_length, 
    nr_reservoir=nr_reservoir,
    spectral_radius=spectral_radius, 
    reg_coefficient=reg_coefficient
)

# Plot the result
plt.figure()
plt.plot(data[train_length:train_length+test_length], label="Ground truth")
plt.plot(Y.flatten(), label="ESN prediction")
plt.legend()
plt.title('ESN Model Prediction vs Ground Truth for Sinusoidal Signal')
plt.show()

print(f"Mean Squared Error for the sinusoidal signal: {mse}")