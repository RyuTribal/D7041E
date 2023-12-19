import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg

class EchoStateNetworkAdapted:
    def __init__(self, data, nr_reservoir, spectral_radius, input_scaling, reg_coefficient, seed=42):
        """
        Initialize the Echo State Network (ESN) with given parameters.
        """
        self.nr_reservoir = nr_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.reg_coefficient = reg_coefficient
        self.data = data
        self.Nu = 1

        np.random.seed(seed)
        self.Win = np.random.uniform(-1, 1, (nr_reservoir, self.Nu + 1)) * 0.2
        self.W = np.random.uniform(-1, 1, (nr_reservoir, nr_reservoir))
        rhoW = max(abs(linalg.eig(self.W)[0]))
        self.W *= spectral_radius / rhoW

    def train(self, first_steps=1000):
        """
        Train the Echo State Network (ESN).
        """
        X = np.zeros((1 + self.Nu + self.nr_reservoir, len(data) - first_steps))
        Yt = data[None, first_steps:]

        self.states = np.zeros((self.nr_reservoir, 1))
        for t in range(len(data)):
            u = data[t]
            self.states = np.tanh(np.dot(self.Win, np.vstack((1, u))) + np.dot(self.W, self.states))
            if t >= first_steps:
                X[:, t - first_steps] = np.vstack((1, u, self.states))[:, 0]

        self.Wout = linalg.solve(np.dot(X, X.T) + self.reg_coefficient * np.eye(1 + self.Nu + self.nr_reservoir), np.dot(X, Yt.T)).T

    def predict(self, test_data):
        """
        Predict using the trained Echo State Network (ESN).
        """
        u = test_data.copy()
        self.predictions = np.zeros((1, len(u)))
        for t in range(len(u)):
            # Reshape the scalar 1 to have the same number of rows as u
            one_vector = np.ones((1, 1))
            u_vector = u[t].reshape(1, -1)  # Reshape u to a 2D array with 1 row

            # Stack the reshaped 1 and u
            stacked_input = np.vstack((one_vector, u_vector))

            # Update states and compute output
            self.states = np.tanh(np.dot(self.Win, stacked_input) + np.dot(self.W, self.states))
            y = np.dot(self.Wout, np.vstack((one_vector, u_vector, self.states)))

            # Store predictions and update u
            self.predictions[:, t] = y
            u[t] = y

        self.mse = sum(np.square(test_data - self.predictions[0, :])) / len(test_data)
        return self.predictions


# Example usage
data = np.sin(np.arange(1, 4001) / 4) * 0.5  # Generate data
esn_adapted = EchoStateNetworkAdapted(data=data[:3000], nr_reservoir=1000, spectral_radius=0.8, input_scaling=0.2, reg_coefficient=1e-8)
esn_adapted.train(first_steps=1000)
predictions_adapted = esn_adapted.predict(data[3000:])

# Compute and print MSE
print("Adapted MSE:", esn_adapted.mse)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(data[3000:], label="Ground Truth")
plt.plot(predictions_adapted[0, :], label="Adapted Predictions")
plt.xlabel("Time Step")
plt.ylabel("Signal Value")
plt.title("Adapted Echo State Network Predictions")
plt.legend()
plt.show()

