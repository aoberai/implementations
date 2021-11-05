'''
Multivariate Kalman Filter
'''

import numpy as np

class KalmanFilter:
    def __init__(self, F, x0, B, u, P, Q, H, R):
        '''
        All matrices are numpy arrays
        @param F: state transition function; x@F is prior (prediction)
        @param x0: initial state mean
        @param B: control input model (factors u into prediction)
        @param u: control input (signal control on plant)
        @param P: initial state covariance; uncertainties in readings
        @param Q: process covariance; uncertainties in state equations
        @param H: measurement function; converts from prediction unit to measurement
        @param R: measurement covariance matrix; measurement noise; dim (mxm | m = sensor count
        I is identity matrix of state var dims
        '''

        self.F = F
        self.x = x0
        self.B = B
        self.u = u
        self.P = P
        self.Q = Q
        self.H = H
        self.R = R
        self.x_prior = None
        self.P_prior = None
        self.I = np.eye(len(x0))
    def predict(self):
        '''
        computes mean and covariance of prior
        '''
        self.x_prior = np.dot(self.F, self.x) + np.dot(self.B, self.u)
        self.P_prior = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return (self.x_prior, self.P_prior)

    def update(self, z):
        '''
        Merges prior and Z to get posterior (state estimation)
        @param z: measurement
        '''
        self.y = z - np.dot(self.H, self.x_prior) # residual
        self.K = np.dot(np.dot(self.P_prior, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.P_prior), self.H.T) + self.R)) # Kalman Gain
        self.x = self.x_prior + np.dot(self.K, self.y) # merge prior prediction and z based on K weighting
        self.P = np.dot(self.I - np.dot(self.K, self.H), self.P_prior)
        return (self.x, self.P)

'''
"
update residual in measurement space not state space because measurements are not invertible. it is not possible to convert a measurement of position into a state containing velocity however you can convert a state containing position and velocity into a equivalent measurement containing only position"

a_var = [70, 80, 90, 100, 110, 120, 130]
b_var = [7, 8, 9, 10, 11, 12, 13]
print(np.cov(a_var, b_var))

W = [70.1, 91.2, 59.5, 93.2, 53.5]
H = [1.8, 2.0, 1.7, 1.9, 1.6]
print(np.cov(H, W))

Some notes:

MultiVariate Gaussian Distributions


mu is n dimensional vector
covariance (short for correlated variances) is n by n
    - diagonal contains variance for each variable
    - ex. sigma_13 is covariance between first and third variables

Covariance matrix is symmetric as sigma_13 = sigma_31

Ex.

array([[  0.025,   2.727],
       [  2.727, 327.235]])

This tells me a_var has variance of 0.025, b_var has variance of 327.235, and a_var and b_var are positively correlated at 2.727: as a_var increases, b_var does too"

Ex2. 

X = np.linspace(1, 10, 100)
Y = np.linspace(1, 10, 100)
np.cov(X, Y)

array([[6.956, 6.956],
       [6.956, 6.956]])

Covariance is equal to variance in x and y. Thus is perfect line

Ex3.
X = np.linspace(1, 10, 100)
Y = -(np.linspace(1, 5, 100) + np.sin(X)*.2)
plot_correlated_data(X, Y)
print(np.cov(X, Y))

[[ 6.956 -3.084]
 [-3.084  1.387]]

covariance sigma_xy is -3.08
x, y slope downward and have strong correlation since far from 0

"Joint probability (P(x, y)) is the probability of both events x and y hapening"
"Marginal probability is the probability of an event happening without regard of any other event"

The marginal of a multivariate gaussian is also gaussian (obvious)

[[2 0]
 [0  2]]

slice is perfect circle

[[2 0]
 [0  6]]
 eliptical slice

[[2 1.2]
 [1.2  2]]
 eliptical tilted right slice

 Any slice through multivarite gaussian is ellipse (known as error ellipses or confidence ellipses)

 Going downward represents standard deviations away from mu

 Covariance ellipse shows oyu how data is scattered in relation to each other; narrower means more correlated


 Extra:

 Pearson Correlation Coefficient defined as

 p_xy = COV(X, Y) / (sigma_x * sigma_y) ; ranges from -1 to 1
    covariance 0 then p = 9
    correlation and covariance are closely related

Independent variables always uncorrelated

Any correlated variables put together make for better prediction

overlap of confidence ellipses through multiplication of gaussian distribution
'''


