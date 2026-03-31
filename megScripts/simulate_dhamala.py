import numpy as np
from spectral_connectivity import Multitaper, Connectivity

# Simulated time-series: 2 regions, 10 dipoles each, 50 epochs, 512 samples
np.random.seed(0)
# Option 1: PCA pooling
data1 = np.random.randn(50, 512, 10)
data2 = np.random.randn(50, 512, 10)

# PCA
from sklearn.decomposition import PCA
pca1 = np.zeros((50, 512))
pca2 = np.zeros((50, 512))
for e in range(50):
    pca1[e] = PCA(n_components=1).fit_transform(data1[e])[:, 0]
    pca2[e] = PCA(n_components=1).fit_transform(data2[e])[:, 0]

data_pca = np.stack([pca1, pca2], axis=2) # (50, 512, 2)
m = Multitaper(time_series=data_pca, sampling_frequency=512)
c = Connectivity.from_multitaper(m)
print("PCA GC:", c.pairwise_spectral_granger_prediction()[0,0,0,:,0])
