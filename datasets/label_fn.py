import pandas as pd

from typing import List


class Key_Labeler:
    def __init__(self, lab: str = 'scene') -> None:
        self.map_ = {}
        self.num_classes = 0
        self.lab = lab

    def __call__(self, df: pd.DataFrame) -> List[int]:
        assert self.lab in df.columns, f'No {self.lab} in meta_data'
        for s in df[self.lab].unique():  # preserve order
            if s not in self.map_:
                self.map_[s] = self.num_classes
                self.num_classes += 1
        labels = df[self.lab].apply(lambda x: self.map_[x]).to_list()
        return labels
