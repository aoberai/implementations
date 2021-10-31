'''
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

https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/04-One-Dimensional-Kalman-Filters.ipynb
'''
import numpy as np
from utils import GaussianDistribution

gd = GaussianDistribution()

'''

Some notes:

posterior is distribution after incorporating the measurement information (after update)
prior is before including measurement's information (predictions)
likelihood is how likely each position is given a measurement - not a probability distribution since does not sum to one
posterior = likelihood * prior / normalization for Bayes Discrete Filter

State after performing prediction the prior or prediction
State after update the posterior or estimated state

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
Ex. To compute how likely it is to rain given specific sensor readings can be computed using the likelihood of the sensor readings given that it is raining
'''
