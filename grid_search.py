import os
import argparse

import torch

from main import main

grid = {}
grid['model_type'] = ['CNN', 'RNN']
grid['learning_rate'] = [3e-4, 1e-4, 5e-5]
grid['batch_size'] = [32, 64, 128]
grid['max_seq_len'] = [50, 100, 150, 300]
grid['embed_size'] = [150, 300, 600, 900]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default='TextClassifier', help="model name")
    parser.add_argument("--model_type", type=str, default='CNN', help="model type")
    parser.add_argument("--save_model_path", type=str, default='./experiment_result/', help="path to save model")

    parser.add_argument("--dataset_path", type=str, default='./dataset/', help="dataset path")
    parser.add_argument("--dataset_name", type=str, default='SST2', help="dataset name")
    parser.add_argument("--text_column_index", type=int, default=1, help="text column index")
    parser.add_argument("--label_column_index", type=int, default=0, help="label column index")
    parser.add_argument("--valid_split_ratio", type=float, default=0.1, help="valid split ratio")

    parser.add_argument("--embed_size", type=int, default=600, help="embedding size")
    parser.add_argument("--rnn_layer_num", type=int, default=1, help="number of rnn layers")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")

    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="number of workers")
    parser.add_argument("--epoch", type=int, default=100, help="epoch")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="early stopping patience")
    parser.add_argument("--max_seq_len", type=int, default=300, help="max sequence length")

    parser.add_argument('--log_interval', type=int, default=500,
                        help='Interval for printing batch loss')
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="use tensorboard")
    parser.add_argument("--tensorboard_log_dir", type=str, default='./tb_logs', help="tensorboard log dir")
    parser.add_argument('--show_all_tensor', type=bool, default=False,
                        help='torch.set_printoptions(profile="full") if True')
    parser.add_argument('--set_detect_anomaly', type=bool, default=False,
                        help='torch.autograd.set_detect_anomaly(True) if True')
    
    args = parser.parse_args()

    args.train_data_path = os.path.join(args.dataset_path, args.dataset_name, 'train.csv')
    args.test_data_path = os.path.join(args.dataset_path, args.dataset_name, 'test.csv')

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    if args.show_all_tensor:
        torch.set_printoptions(profile="full")
    if args.set_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for each_model_type in grid['model_type']:
        for each_learning_rate in grid['learning_rate']:
            for each_batch_size in grid['batch_size']:
                for each_max_seq_len in grid['max_seq_len']:
                    for each_embed_size in grid['embed_size']:
                        args.model_type = each_model_type
                        args.learning_rate = each_learning_rate
                        args.batch_size = each_batch_size
                        args.max_seq_len = each_max_seq_len
                        args.embed_size = each_embed_size

                        main(args)
    