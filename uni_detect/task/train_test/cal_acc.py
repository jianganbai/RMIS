import os
import copy
import pandas as pd

from typing import Tuple, Dict, Any


class Train_Test_Evaluator:
    '''
        Evaluator for explicit train test split
    '''
    def __init__(self, args: Dict[str, Any]) -> None:
        self.CLS_MAP = copy.deepcopy(args['map_'])
        self.INV_CLS_MAP = {v: k for k, v in self.CLS_MAP.items()}

    def cal_acc(
        self,
        pred_df: pd.DataFrame,
        exp_dir: str
    ) -> Tuple[float, Dict[str, Any]]:
        lab_set = sorted(pred_df['label'].unique())

        pred_df['cls'] = pred_df['label'].apply(lambda x: self.INV_CLS_MAP[x])
        pred_df['eval'] = pred_df.apply(lambda x: int(x['label'] == x['pred']), axis=1)

        result_df = pd.DataFrame({'cls': [], 'pred': [], 'correct': [], 'all': [], 'acc': []})
        num_acc, num_all = 0, 0
        for lab in lab_set:
            cls_bool = pred_df['label'] == lab
            num_cls_cor = pred_df.loc[cls_bool, 'eval'].sum()
            num_cls = cls_bool.sum()
            result_df.loc[len(result_df)] = [
                self.INV_CLS_MAP[lab],
                (pred_df['pred'] == lab).sum(),
                num_cls_cor,
                num_cls,
                num_cls_cor / num_cls * 100
            ]
            num_acc += num_cls_cor
            num_all += num_cls
        macro_acc = result_df['acc'].mean()
        micro_acc = num_acc / num_all * 100
        result_df['acc'] = result_df['acc'].map(lambda x: round(x, 2))

        pred_df['pred'] = pred_df['pred'].apply(lambda x: self.INV_CLS_MAP[x])
        pred_df.drop('label', axis=1, inplace=True)
        pred_df.to_csv(os.path.join(exp_dir, 'pred.csv'), index=False)

        with open(os.path.join(exp_dir, 'acc.csv'), 'w') as fp:
            fp.write(result_df.__repr__() + '\n')
            fp.write('-' * 35 + '\n\n')
            fp.write(f'macro-acc: {macro_acc:.2f}\n')
            fp.write(f'micro-acc: {micro_acc:.2f}\n')
        return macro_acc, {'macro-acc': macro_acc}
