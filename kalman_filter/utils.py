import numpy as np
import matplotlib.pyplot as plt
import math

class Utils:
    def get_data(self, n, x0, v, dt, mu, sigma, a=0):
        return [x0 + 0.5 * a * (i * dt) ** 2 + v * dt * i + sigma * np.random.randn() + mu for i in range(n)]

    def plot(self, y_set, dt, x_0=0):
        series = []
        for y in range(len(y_set.values())):
            series_val = list(y_set.values())[y]
            series.append(plt.scatter(
                np.array([i * dt + x_0 for i in range(len(series_val))]), np.array(series_val), 5))
        plt.legend(series, y_set.keys(), fontsize=8, loc='upper left')
        plt.show()

class GaussianDistribution:
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        self.var = sigma**2 # was added later, thus not nicely integrated

    def pdf(self, x):
        return (math.e ** (-0.5 * ((x - self.mu)/self.sigma)**2))/((math.pi*2)**0.5*self.sigma)
    def cdf(self, x0, x1):
        # TODO: Implement
        return 0
    def scale(self, k):
        self.mu *= k
        return self
    def shift(self, k):
        self.mu += k
        return self
    def to_string(self):
        return "Mu: %0.3f; Sigma: %0.3f" % (self.mu, self.sigma)
    @staticmethod
    def add(gd1, gd2):
        return GaussianDistribution(gd1.mu + gd2.mu, (gd1.sigma**2 + gd2.sigma**2)**0.5)
    @staticmethod
    def add_list(gd_list):
        return GaussianDistribution(np.sum([gd.mu for gd in gd_list]), np.sum([gd.sigma**2 for gd in gd_list])**0.5)
    @staticmethod
    def multiply(gd1, gd2):
        return GaussianDistribution((gd1.sigma**2*gd2.mu + gd2.sigma**2*gd1.mu)/(gd1.sigma**2+gd2.sigma**2), ((gd1.sigma**2)*(gd2.sigma**2))/(gd1.sigma**2+gd2.sigma**2))




