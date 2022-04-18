# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

# @Time   : 2022/4/8
# @Author : SUyeon Hong 

import argparse
import os

from start import run_recbole_fdsa

parameter_dict = {
    'neg_sampling': None,
    'user_inter_num_interval': "[30,inf)",
    'item_inter_num_interval': "[40,inf)",
}

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, 'config/Config_fdsa.yaml')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SINE', help='name of models')  # 'BPR', 'fdsa'
    parser.add_argument('--dataset', '-d', type=str, default='ml_bc' , help='name of datasets') # 'ml-100k' 'ml_bc'
    parser.add_argument('--config_files', type=str, default=None, help='config files')


    args, _ = parser.parse_known_args()

    # config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_fdsa(model=args.model, dataset=args.dataset, config_file_list= config_file_list, config_dict = parameter_dict)
