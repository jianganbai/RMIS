import os
import argparse

from rmis.scripts.reg_all import rmis_stat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='fisher')
    parser.add_argument('--rel_exp_dir', type=str, default='small')
    args = parser.parse_args()
    rmis_out_dir = os.path.join('exp', args.model_name, 'rmis/fix', args.rel_exp_dir)
    os.makedirs(rmis_out_dir, exist_ok=True)
    rmis_stat(args.model_name, args.rel_exp_dir, rmis_out_dir)
