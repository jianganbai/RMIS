import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA as PCAs

from typing import Dict, Any, Optional


class KNN_Detector_SKLearn:
    def __init__(
        self,
        n_neighbors: int = 1,
        metric: str = 'cosine',
        score_train: bool = False,
        pca_n_components: int = -1,
        **kwargs
    ) -> None:
        self.n_neighbors = n_neighbors
        self.clf = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric
        )
        self.score_train = score_train
        if pca_n_components > 0:
            self.pca = PCAs(n_components=pca_n_components)

    def score(
        self,
        embs_test: np.ndarray,
        embs_train: np.ndarray,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        scores = {}
        if hasattr(self, 'pca'):
            self.pca.fit(embs_train)
            embs_train = self.pca.transform(embs_train)
            embs_test = self.pca.transform(embs_test)
        self.clf.fit(embs_train)
        scores['test'] = self.clf.kneighbors(embs_test, self.n_neighbors)[0].mean(axis=-1)
        if self.score_train:
            scores['train'] = self.clf.kneighbors(embs_train, self.n_neighbors)[0].mean(axis=-1)
        return scores


class Anomaly_Detector:
    def __init__(
        self,
        conf: Dict[Any, Any],
        prefix: Optional[str] = None,
    ) -> None:
        self.map_detector = {
            'knn': KNN_Detector_SKLearn,
        }
        self.use_detectors = conf['detector']
        self.prefix = prefix
        self.build_detectors(conf)

    def build_detectors(self, conf: Dict[Any, Any]) -> None:
        self.detector = {}
        self.dtr_conf = {}
        for d in self.use_detectors:
            assert d in self.map_detector.keys(), f'Detector {d} is not implemented!'
            if self.prefix:
                self.dtr_conf[d] = conf[f'{d}_conf'].get(self.prefix, {})
            else:
                self.dtr_conf[d] = conf.get(f'{d}_conf', {})
            self.detector[d] = self.map_detector[d](**self.dtr_conf[d])

    def score(
        self,
        embs: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        '''conduct anomaly detection

        Args:
            embs (Dict[str, np.ndarray]): {'embs_test': np.ndarray, 'embs_train': np.ndarray, **kwargs}

        Returns:
            Dict[str, Dict[str, np.ndarray]]: {'detector_name': {'test': np.ndarray}}
        '''
        scores_dict = {}
        for d, dd in self.detector.items():
            scores = dd.score(**embs)
            if self.dtr_conf[d].get('normalize', True):
                for k in scores.keys():
                    scores[k] = (scores[k] - np.mean(scores[k])) / np.std(scores[k])
            scores_dict[d] = scores
        return scores_dict
