'''
Multivariate Kalman Filter
'''

import numpy as np
import utils as util

class KalmanFilter:
    def __init__(self, F, x0, B, P, Q, H, R, dt):
        '''
        All matrices are numpy arrays
        @param F: state transition function; x@F is prior (prediction); states x states
        @param x0: initial state mean; states x 1
        @param B: control input model (factors u into prediction); states x inputs
        @param u: control input (signal control on plant); inputs x 1
        @param P: state covariance; uncertainties in readings; states x states
        @param Q: process covariance; uncertainties in state equations; states x states
        @param H: measurement function; converts from prediction unit to measurement; outputs x states
        @param R: measurement covariance matrix; measurement noise; outputs x outputs
        I is identity matrix of state var dims
        * Diff literature denotates some differently; F aka A aka system matrix; H aka C aka output matrix; B aka input matrix; D aka Feedthrough Matrix (not implemented) etc.
        '''

        self.F = F
        self.x = x0
        self.B = B
        self.P = P
        self.Q = Q
        self.H = H
        self.R = R
        self.x_prior = None
        self.P_prior = None
        self.I = np.eye(len(x0))
        self.dt = dt

    def predict(self, u):
        '''
        computes mean and covariance of prior

        xbar = Fx + Bu
        Pbar = FPF.T + Q
        '''
        self.x_prior = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P_prior = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return (self.x_prior, self.P_prior)

    def update(self, z):
        '''
        Merges prior and Z to get posterior (state estimation)
        @param z: measurement

        y = z - Hxbar
        K = Pbar*H.T*inv(H*Pbar*H.T + R)
        x = xbar + Ky
        P = (I - KH)*Pbar
        '''
        self.y = z - np.dot(self.H, self.x_prior) # residual (outputs x 1)
        self.K = np.dot(np.dot(self.P_prior, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.P_prior), self.H.T) + self.R)) # Kalman Gain
        self.x = self.x_prior + np.dot(self.K, self.y) # merge prior prediction and z based on K weighting
        self.P = np.dot(self.I - np.dot(self.K, self.H), self.P_prior)
        return (self.x, self.P)
    def get_K(self):
        '''
        @return K: states x outputs
        '''
        return self.K

if __name__ == '__main__':
    dt = 1
    x = np.array([x0:= 0, v0:= 3])
    P = np.eye(2)
    F = np.array([
                    [1, dt], [0, 1]
                 ])
    B = 0
    u = 0
    Q = np.eye(2)
    H = np.array([1, 0]).reshape((1, 2))

    z = np.array([x0])
    R = np.array([measure_var:=1])

    kf = KalmanFilter(F=F, x0=x, B=B, P=P, Q=Q, H=H, R=R, dt=dt)


    # General Test


    utils = util.Utils()
    z_positions = utils.get_data(
        n=100, x0=x0, v=v0, a=0, dt=dt, mu=0, sigma=10)
    prior_positions = []
    posterior_positions = []
    ks = []

    for z in z_positions:
        prior = kf.predict(u)
        posterior = kf.update(z)
        prior_positions.append(prior)
        posterior_positions.append(posterior)
        print("K:", kf.K)
    utils.plot({"predicted": [r[0][0] for r in prior_positions]
                , "observed": z_positions,
               "estimated": [r[0][0] for r in posterior_positions]}, dt)

