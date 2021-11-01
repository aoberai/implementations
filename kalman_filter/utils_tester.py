import utils
from utils import GaussianDistribution

gd = GaussianDistribution(0, 1)
ut = utils.Utils()
ut.plot({"Standard Normal: N(0, 1)": [gd.pdf(i) for i in range(-4, 4)]}, 1, -4)
gd = GaussianDistribution(0, 2)
ut.plot({"Standard Normal: N(0, 2)": [gd.pdf(i) for i in range(-4, 4)]}, 1, -4)

print(GaussianDistribution.multiply(gd, gd).to_string())
