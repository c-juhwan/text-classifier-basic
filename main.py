import os
import time
import argparse
from tqdm.auto import tqdm

import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from epoch import train_epoch, valid_epoch, test_model
from model.cnn import ClassifierCNN
from model.rnn import ClassifierRNN
from utils.dataset import CustomDataset, build_dataset
from utils.vocab import build_vocab
from utils.util import set_random_seed, get_tb_experiment_name

def main(args):
    set_random_seed(args.seed)
    ts = time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocabulary = build_vocab(args)

    datasets = build_dataset(args, vocabulary)
    dataloaders = {}
    for each in ['train', 'valid']:
        dataloaders[each] = torch.utils.data.DataLoader(datasets[each], batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=args.num_workers)
    dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.model_type == 'CNN':
        model = ClassifierCNN(len(vocabulary), args.embed_size, args.num_classes)
    elif args.model_type == 'RNN':
        model = ClassifierRNN(len(vocabulary), args.embed_size, args.num_classes, args.rnn_layer_num)
    else:
        raise NotImplementedError('Model type {} is not implemented'.format(args.model_type))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    if args.use_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_log_dir, get_tb_experiment_name(args, ts)))
        writer.add_text('model', str(model))
        writer.add_text('args', str(args))
        writer.add_text('ts', ts)

    for epoch_idx in range(0, args.epoch):
        train_epoch(args, epoch_idx, model, dataloaders['train'], optimizer, loss_fn, writer, device)
        if valid_epoch(args, epoch_idx, model, dataloaders['valid'], optimizer, loss_fn, writer, device) is False:
            break
    test_model(args, model, dataloaders['test'], loss_fn, writer, device)

    save_model_name = f'{args.model_name}_{ts}.pt'
    torch.save(model.state_dict(), os.path.join(args.save_model_path, save_model_name))

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

    main(args)