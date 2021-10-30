'''
Make prediction -> using state estimation formula
Get noisy sensor data
Merge prediction and sensor data


**Initialization**

    1. Initialize the state of the filter
    2. Initialize our belief in the state

**Predict**

    1. Use system behavior to predict state at the next time step
    2. Adjust belief to account for the uncertainty in prediction

**Update**

    1. Get a measurement and associated belief about its accuracy
    2. Compute residual between estimated state and measurement
    3. New estimate is somewhere on the residual line

https://hub-binder.mybinder.ovh/user/rlabbe-kalman-a-lters-in-python-7284y12e/lab/tree/01-g-h-filter.ipynb
'''

import numpy as np
import matplotlib.pyplot as plt

class GHFilter:
    """
    Performs g-h filter on 1 state variable with a fixed g and h.
    'data' contains the data to be filtered.
    'x0' is the initial value for our state variable
    'v' is the change rate for our state variable
    'g' is the g-h's g scale factor 'h' is the g-h's h scale factor
    'dt' is the length of the time step

    'z' is measured value
    'x' is estimated position
    'k' denotes timestep

    'p' denotes plus; 'm' denotes minus
    """
    def __init__(self, x0, v, g, h, dt):
        self.x0 = self.x = x0
        self.v = v
        self.g = g
        self.h = h
        self.dt = dt
    def predict(self):
        self.x_pred = self.v * self.dt + self.x
        # TODO: predict both future pos and future gain rate
        return self.x_pred
    def update(self, z):
        resid = z - self.x_pred
        # updates x estimation
        self.x = self.x_pred + self.g*resid
        # updates change in model for next timestep prediction 
        self.v += self.h*resid/self.dt
        return self.x

class Utils:
    def get_data(self, n, x0, v, dt, mu, sigma, a=0):
        return [x0 + 0.5 * a * (i * dt) ** 2 + v*dt*i + sigma * np.random.randn() + mu for i in range(n)]
    def plot(self, y_set, dt):
        series = []
        for y in range(len(y_set.values())):
            series_val = list(y_set.values())[y]
            series.append(plt.scatter(np.array([i*dt for i in range(len(series_val))]), np.array(series_val), 5))
        plt.legend(series, y_set.keys(), fontsize=8, loc='upper left')
        plt.show()

if __name__ == "__main__":
    '''
     Larger g -> follow measurement more than prediction
     Smaller g -> follow prediction more than measurement
     Optimize g such that can tell change in measurement due to unexpected force (not in model) while reducing noise through following more stable prediction

     Larger h -> follow measurement of v more than prediction
     Smaller g -> follow the measurement of v less than prediction

     If signal changing rapidly, want smaller g (relies on measurement more than prediction)
     If signal changing slowly, want larger g (relies on prediction more than measurement)
    '''
    x0_obs, x0_guess, v, g, h, dt = 5, 100, 2, 0.2, 0.02, 1
    utils = Utils()
    ghfilter = GHFilter(x0_guess, v, g, h, dt)
    # Acceleration show that gh filter can only model constant speed data
    # Need third term k to weight acceleration estimate
    observed_data = utils.get_data(n=10, x0=x0_obs, v=v, dt=1, mu=0, sigma=5, a=10)
    xs_estimated = []
    predicted = []
    for z in observed_data:
        predicted.append(prediction := ghfilter.predict())
        xs_estimated.append(x_estimated := ghfilter.update(z))
    print("Predicted: ", predicted, "\nObserved: ", observed_data, "\nEstimated: ", xs_estimated)
    utils.plot({"predicted": predicted, "observed": observed_data, "estimated": xs_estimated}, dt)





