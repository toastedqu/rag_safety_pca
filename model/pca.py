from typing import Optional

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAClassifier:
    """
    PCA Classifier

    Attributes:
        n_components (int): number of components
        scale (bool): whether to scale the data
        pca (PCA): PCA model

    Args:
        n_components (int): number of components
        scale (bool): whether to scale the data
    """

    def __init__(self, n_components: Optional[int] = None, scale: bool = True):
        self.n_components = n_components
        self.scale = scale
        self.pca = PCA(n_components=n_components)

    def fit(self, embeddings: list) -> PCA:
        if self.scale:
            embeddings = StandardScaler().fit_transform(embeddings)
        self.pca.fit(embeddings)

        return self.pca
