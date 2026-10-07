import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import List, Tuple

def learn_codebook(blocks: List[Tuple], vocab_size: int):
    data = np.stack([b[3] for b in blocks])
    # Normalize for better clustering
    # Simple sampling
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, batch_size=1024, random_state=0)
    kmeans.fit(data)
    codebook = kmeans.cluster_centers_
    return codebook
