import numpy as np

def kl_divergence_loss(mean_vector, log_var_vector):
    # Calculates divergence to N(mu = 0, sigma**2 = 1)
    loss = 0.5 * np.sum(np.exp(0.5 * log_var_vector) + np.square(mean_vector) - 1 - 0.5 * log_var_vector)/len(mean_vector)
    return loss

print(kl_divergence_loss(np.array([0, 0]), np.array([1.38629, 1.38629])))

