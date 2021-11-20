'''

Tracks 1 state variable

Initialization

1. Initialize the state of the filter
2. Initialize our belief in the state

Predict

1. Use system behavior to predict state at the next time step
2. Adjust belief to account for the uncertainty in prediction

Update

1. Get a measurement and associated belief about its accuracy
2. Compute residual between estimated state and measurement
3. Compute scaling factor based on whether the measurement
or prediction is more accurate
4. set state between the prediction and measurement based
on scaling factor
5. update belief in the state based on how certain we are
in the measurement
'''
import numpy as np
import utils as util
from utils import GaussianDistribution
import math

class KalmanFilter1d:
    # TODO: update v and a or not needed?
    def __init__(self, x_0: GaussianDistribution = None, v_0: GaussianDistribution = None, a_0: GaussianDistribution = None):
        self.x = x_0
        self.v = v_0
        self.a = a_0
        self.likelihood = x_0
        self.prior = x_0


    '''
    Traditionally:

    z is measurement
    P is variance of state
    Q is process noise
    R is measurement noise
    '''


    '''
    Constant accel model; can be replaced with other state space model
    '''
    def predict_bayesian(self, dt):
        return GaussianDistribution.add_list([self.x, self.v.scale(dt), self.a.scale(0.5*(dt**2))])

    '''
    Movement is dx prediction; simple linear motion with movement as velocity model; primarily here for variable reference
    '''
    def predict_trad(self, posterior, movement):
        return GaussianDistribution(x:=posterior.mu+movement.mu, P:=(posterior.sigma**2 + (Q:=movement.sigma**2))**0.5)

    '''
    Finding P(A | B) given P(B | A) aka likelihood and P(A) aka prior

    'prior' aka measured position
    'likelihood' aka predicted position

    return 'posterior' aka estimated position
    '''
    def update_bayesian(self, likelihood, prior):
        self.prior = prior
        posterior = self.x = GaussianDistribution.multiply(likelihood, prior)
        return posterior

    '''
    prior (prediction) (Different for bayesian and traditional method)
    measurement
    '''
    def update_trad(self, prior, measurement):
        y = measurement.mu - prior.mu # residual
        K = prior.sigma**2 / (prior.sigma**2 + measurement.sigma**2) # K gain
        x = prior.mu + K*y
        P = (1 - K) * prior.sigma**2 # posterior variance
        self.x = GaussianDistribution(x, P**0.5)
        self.prior = measurement
        return self.x

    def get_k_gain(self):
        return (self.likelihood.sigma**2)/(self.likelihood.sigma**2 + self.prior.sigma**2)

if __name__ == '__main__':
    '''
    One Cycle
    '''

    # prior = GaussianDistribution(10, 0.2) # x0
    # kf_filter = KalmanFilter1d(prior, GaussianDistribution(15, 0.7), GaussianDistribution(0, 0))
    # likelihood = kf_filter.predict(dt=1)
    # posterior = kf_filter.update(likelihood, prior)
    # print("Measured State", prior.to_string(), "Predicted Position", likelihood.to_string(), "Estimated State", posterior.to_string())

    '''
    Loop
    '''

    sensor_var = 0.2
    x0 = GaussianDistribution(10, sensor_var**0.5)
    v0 = GaussianDistribution(5, 0.7)
    a0 = GaussianDistribution(1, 0)
    posterior = None
    # groundtruths
    observed_x0 = 10
    observed_v0 = 5
    observed_a0 = 1
    predicted, x_estimated = [], []
    dt = 1
    utils = util.Utils()
    observed_data = utils.get_data(
        n=100, x0=observed_x0, v=observed_v0, a=observed_a0, dt=dt, mu=0, sigma=75)
    # observed_data = [2 * math.sin(i/3.) for i in range(100)]

    kf_filter = KalmanFilter1d(x0, v0, a0)
    for z in observed_data:
        prior = GaussianDistribution(z, sensor_var**0.5)
        likelihood = kf_filter.predict_bayesian(dt=1)
        posterior = kf_filter.update_bayesian(likelihood, prior)
        print("Measured State", prior.to_string(), "Predicted Position", likelihood.to_string(), "Estimated State", posterior.to_string(), "K Gain", kf_filter.get_k_gain())

        predicted.append(likelihood.mu)
        x_estimated.append(posterior.mu)

    utils.plot({"predicted": predicted, "observed": observed_data,
               "estimated": x_estimated}, dt)
    for i in range(10):
        print()
    predicted.clear()
    x_estimated.clear()

    '''
    Trad Update vs Bayesian Update
    '''
    # kf_filter = KalmanFilter1d(x0, v0, a0)
    # for z in observed_data:
    #     prior = GaussianDistribution(z, sensor_var**0.5)
    #     likelihood = kf_filter.predict_bayesian(dt=1)
    #     posterior = kf_filter.update_trad(prior=likelihood, measurement=prior)
    #     print("Measured State", prior.to_string(), "Predicted Position", likelihood.to_string(), "Estimated State", posterior.to_string(), "K Gain", kf_filter.get_k_gain())
    #
    #     predicted.append(likelihood.mu)
    #     x_estimated.append(posterior.mu)
    #
    # utils.plot({"predicted": predicted, "observed": observed_data,
    #            "estimated": x_estimated}, dt)


