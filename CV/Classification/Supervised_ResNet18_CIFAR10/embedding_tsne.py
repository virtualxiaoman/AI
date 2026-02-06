import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

emb = np.load('../../../Models/CIFAR10/resnet18_minimal/embeddings.npy')
labels = np.load('../../../Models/CIFAR10/resnet18_minimal/labels.npy')

z = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(emb)

plt.figure(figsize=(8,8))
plt.scatter(z[:,0], z[:,1], c=labels, s=5, cmap='tab10')
plt.colorbar()
plt.savefig('tsne_resnet18_val.png', dpi=200)
plt.show()
