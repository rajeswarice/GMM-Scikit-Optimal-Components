import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

img = cv2.imread("test/img_20200611_102834_043.jpg")

img2 = img.reshape((-1,3))

n_components = np.arange(2,10)

gmm_models = [GMM(n, covariance_type='tied').fit(img2) for n in n_components]

plt.plot(n_components, [m.aic(img2) for m in gmm_models], label='AIC')
plt.xlabel('n_components')
plt.show()

plt.plot(n_components, [m.bic(img2) for m in gmm_models], label='BIC')
plt.xlabel('n_components')
plt.show()