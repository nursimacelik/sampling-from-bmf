from geomstats.geometry.stiefel import Stiefel
import numpy as np
import geomstats.visualization as visualization
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.linalg import svd, orth
import tensorflow_probability as tfp
import random


def ratio(Y, N, H):
  """ The ratio of pmf(X)/g(X) to the upper bound K(H) (pg. 6 of the paper). """
  result = 1
  for i in range(1, p):
    nth = np.linalg.norm(N[:,i].dot(H[:,i]))
    h = np.linalg.norm(H[:,i])
    frac = (h / nth)**((n-i-1)//2)
    result *= (np.eye((n-i-1)//2) * nth) / (np.eye((n-i-1)//2) * h) * frac
  return result


def sampler(size=10):
  """ Sample from matrix von MF distribution using rejection sampling. """
  samples = []

  # Singular-value decomposition
  U, D, VT = svd(F)

  for i in range(size):
    # the modal orientation
    H = U * D

    while True:
      # obtain {u, Y} pairs until the acceptance criteria is satisfied

      # sample u from uniform distribution
      u = random.random()
      
      dist = tfp.distributions.VonMisesFisher(H[:,0], D)
      
      # sample Y from von Mises Fisher distribution column by column
      # while making sure that each column is orthogonal to all the previous columns

      Y = np.array(dist.sample())
      for j in range(1, 2):
        N = orth(Y.T)    # N is orthonormal basis for Y
        zDist = tfp.distributions.VonMisesFisher(N.T.dot(H[:,j]), D)
        z = zDist.sample()
        np.append(Y, N.dot(z))
      
      if u < ratio(Y, N, H):
        # accept
        break
      
    X = Y.T.dot(VT)
    samples.append(X)

  # adjust shape for visualization
  samples = np.array(samples)
  if samples.shape[1] == 1:
    samples = np.reshape(samples, (samples.shape[0], samples.shape[2]))
  elif samples.shape[2] == 1:
    samples = np.reshape(samples, (samples.shape[0], samples.shape[1]))
  return samples


# hyperparameters
n = 3
p = 1
stiefel = Stiefel(n, p)    # sphere
F = np.random.rand(n, p)
print("F", F)

data = sampler(500)

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')

visualization.plot(
    data, ax=ax, space='S2', label='Point', s=80)
