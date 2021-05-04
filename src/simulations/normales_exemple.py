import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


# Parametres
N_OBS = 5000

# Normal dist
MU = 0
SIGMA = 2

np.random.seed(666)

# Simulations des données + graphique
x = MU + SIGMA * np.random.randn(N_OBS)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)

print(f"proportion out: {len(np.argwhere(np.abs(x) >= 6))/N_OBS}")

plt.xlabel('Valeurs')
plt.ylabel('Fréquence relative')
plt.xlim(-25, 25)
plt.ylim(0, 0.25)
plt.grid(True)
plt.savefig("histogram-normal-ztest.pdf")
plt.clf()


# Simulations des gammas
chy = np.random.standard_cauchy(N_OBS)

# the histogram of the data
n, bins, patches = plt.hist(chy[np.abs(chy)<50], 50, density=True, facecolor='g', alpha=0.75)

print(f"proportion out: {len(np.argwhere(np.abs(chy) >= (np.mean(chy) + 3 * np.std(chy))))/N_OBS}")
print(f"mean: {np.mean(chy)}")
print(f"sd: {np.std(chy)}")

plt.xlabel('Valeurs')
plt.ylabel('Fréquence relative')
plt.xlim(-25, 25)
plt.ylim(0, 0.3)
plt.grid(True)
plt.savefig("histogram-cauchy-ztest.pdf")