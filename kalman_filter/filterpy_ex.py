import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import utils as util

dt = 1
x0, v0, a0 = 1, 0, 0
my_filter = KalmanFilter(dim_x=2, dim_z=1)

my_filter.x = np.array([[2.],
                        [0.]])       # initial state (location and velocity)

my_filter.F = np.array([[1.,1.],
                        [0.,1.]])    # state transition matrix

my_filter.H = np.array([[1.,0.]])    # Measurement function
my_filter.P *= 1000.                 # covariance matrix
my_filter.R = 5                      # state uncertainty
my_filter.Q = Q_discrete_white_noise(2, dt, .1) # process uncertainty

utils = util.Utils()
observed_data = utils.get_data(
    n=100, x0=x0, v=v0, a=a0, dt=dt, mu=0, sigma=100)

for i in observed_data:
    my_filter.predict()
    my_filter.update(i)

    # do something with the output
    x = my_filter.x
    print(x)
