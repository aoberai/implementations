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

class KalmanFilter1d:
    def __init__(self, x_0: GaussianDistribution, v_0: GaussianDistribution, a_0: GaussianDistribution):
        self.x = x_0
        self.v = v_0
        self.a = a_0

    '''
    Constant accel model; can be replaced with other state space model
    '''
    def predict(self, dt):
        return GaussianDistribution.add_list([self.x, self.v.scale(dt), self.a.scale(0.5*(dt**2))])

    '''
    Finding P(A | B) given P(B | A) aka likelihood and P(A) aka prior

    'prior' aka measured position
    'likelihood' aka predicted position

    return 'posterior' aka estimated position
    '''
    def update(self, likelihood, prior):
        posterior = self.x = GaussianDistribution.multiply(likelihood, prior)
        return posterior

if __name__ == '__main__':
    '''
    One Cycle
    '''

    prior = GaussianDistribution(10, 0.2) # x0
    kf_filter = KalmanFilter1d(prior, GaussianDistribution(15, 0.7), GaussianDistribution(0, 0))
    likelihood = kf_filter.predict(dt=1)
    posterior = kf_filter.update(likelihood, prior)
    print("Measured State", prior.to_string(), "Predicted Position", likelihood.to_string(), "Estimated State", posterior.to_string())

    '''
    Loop
    '''

    sensor_var = 0.2
    x0 = GaussianDistribution(10, sensor_var**0.5)
    v0 = GaussianDistribution(15, 0.7)
    a0 = GaussianDistribution(0, 0)
    predicted, x_estimated = [], []
    dt = 1
    utils = util.Utils()
    observed_data = utils.get_data(
        n=10, x0=x0.mu, v=v0.mu, a=a0.mu, dt=dt, mu=0, sigma=5)
    kf_filter = KalmanFilter1d(x0, v0, a0)
    for z in observed_data:
        prior = GaussianDistribution(z, sensor_var**0.5)
        likelihood = kf_filter.predict(dt=1)
        posterior = kf_filter.update(likelihood, prior)
        print("Measured State", prior.to_string(), "Predicted Position", likelihood.to_string(), "Estimated State", posterior.to_string())

        predicted.append(likelihood.mu)
        x_estimated.append(posterior.mu)

    utils.plot({"predicted": predicted, "observed": observed_data,
               "estimated": x_estimated}, dt)
'''

Some notes:

posterior is distribution after incorporating the measurement information (after update)
prior is before including measurement's information (predictions)
likelihood is how likely each position is given a measurement - not a probability distribution since does not sum to one
posterior = likelihood * prior / normalization for Bayes Discrete Filter

State after performing prediction the prior or prediction
State after update the posterior or estimated state

Estimated state is basically just prior (measured state) times likelihood (predicted state); by nature of Gaussian Distribution, creates a more accurate estimate (narrower product gaussian)

Bayes Rule:
              ^
         A  /    |  !A
           /      |
          /        |
         /          |
        / |         /  |
     B /   | !B    / B  | !B

B through A -> P(B | A)
B through !A -> P(B | !A)

Conversely, P(A | B) is P (A U B) / P(B) (probability of both over given)
    - P(B) is P(A) * P(B | A) + P(!A) * P(B | !A)
    - P(A U B) is P(A) * P(B|A)

Thus, Bayes Theorem is P(A | B):
        P(A) * P(B|A)
   ------------------------
P(A) * P(B | A) + P(!A) * P(B | !A)

            OR

        P(A) * P(B|A)
   ------------------------
            P(B)

Also works for probability distribution ^

----
B is evidence
p(A) is the prior -> belief before incorporating measurements
p(B | A) is the likelihood -> probability type
p(A | B) is the posterior


Update function is just bayes rule (do more research here)
Predict function is total probability theorem; predict computes probability of being at any given position given the probability of all the possible movement events
Prediction is just kinematics
Ex. To compute how likely it is to rain given specific sensor readings can be computed using the likelihood of the sensor readings given that it is raining
'''
