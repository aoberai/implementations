'''
Multivariate Kalman Filter
'''

import numpy as np

a_var = [70, 80, 90, 100, 110, 120, 130]
b_var = [7, 8, 9, 10, 11, 12, 13]
print(np.cov(a_var, b_var))

W = [70.1, 91.2, 59.5, 93.2, 53.5]
H = [1.8, 2.0, 1.7, 1.9, 1.6]
print(np.cov(H, W))
'''
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

