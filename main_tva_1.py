import torch
import numpy as np
import argparse
from data_utils import *
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset
import train_tva_1
import random

if __name__ == '__main__':
    # get arguments
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--data_path', type=str, default='/processed_Yoon/IEMOCAP/seven_category_120/folds/fold01')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--rnntype', type=str, default='gru')
    p.add_argument('--rnndir', type=str, default=True,
                   help='Uni (False) or Bi (True) directional')
    p.add_argument('--rnnsize', type=int, default=60)#30)#200
    # video params
    p.add_argument('--vid_rnnnum', type=int, default=1)#1)#3
    p.add_argument('--vid_rnndp', type=int, default=0.3)#0.3
    p.add_argument('--vid_rnnsize', type=int, default=60)
    p.add_argument('--vid_nh', type=int, default=6,
                        help='number of attention heads for mha')#4
    p.add_argument('--vid_dp', type=int, default=0.1,
                        help='dropout rate for mha')#0.1
    # text params
    p.add_argument('--txt_rnnnum', type=int, default=1)
    p.add_argument('--txt_rnndp', type=int, default=0.3)#0.3
    p.add_argument('--txt_rnnsize', type=int, default=60)
    p.add_argument('--txt_nh', type=int, default=6,
                   help='number of attention heads for mha')#4
    p.add_argument('--txt_dp', type=int, default=0.1,
                   help='dropout rate for mha')#0.1
    # audio params
    p.add_argument('--aud_rnnnum', type=int, default=1)
    p.add_argument('--aud_rnndp', type=int, default=0.3)  # 0.3
    p.add_argument('--aud_rnnsize', type=int, default=60)
    p.add_argument('--aud_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--aud_dp', type=int, default=0.1,
                   help='dropout rate for mha')  # 0.1
    # tv params
    p.add_argument('--tv_nh', type=int, default=6,
                   help='number of attention heads for mha')#4
    p.add_argument('--tv_dp', type=int, default=0.1,
                   help='dropout rate for mha')#0.1
    # ta params
    p.add_argument('--ta_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--ta_dp', type=int, default=0.1,
                   help='dropout rate for mha')  # 0.1
    # vt params
    p.add_argument('--vt_nh', type=int, default=6,
                   help='number of attention heads for mha')#4
    p.add_argument('--vt_dp', type=int, default=0.1,
                   help='dropout rate for mha')
    # va params
    p.add_argument('--va_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--va_dp', type=int, default=0.1,
                   help='dropout rate for mha')
    # at params
    p.add_argument('--at_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--at_dp', type=int, default=0.1,
                   help='dropout rate for mha')
    # av params
    p.add_argument('--av_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--av_dp', type=int, default=0.1,
                   help='dropout rate for mha')

    # tf params
    p.add_argument('--tf_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--tf_dp', type=int, default=0.1,
                   help='dropout rate for mha')

    # vf params
    p.add_argument('--vf_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--vf_dp', type=int, default=0.1,
                   help='dropout rate for mha')

    # af params
    p.add_argument('--af_nh', type=int, default=6,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--af_dp', type=int, default=0.1,
                   help='dropout rate for mha')

    p.add_argument('--output_dim', type=int, default=7,
                        help='number of classes')
    p.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    params = p.parse_args()
    #seed = 123
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(params.seed)

    # get train data
    from sklearn.preprocessing import StandardScaler
    scaler_mfcc = StandardScaler()
    x_text, x_vid, vid_seq, x_mfcc, x_pros, aud_seq, labels = get_text_video_audio_data(params.data_path, 'train')
    s1 = x_mfcc.shape[1]
    s2 = x_mfcc.shape[2]
    x_mfcc = np.reshape(x_mfcc, [x_mfcc.shape[0], -1])
    scaler_mfcc.fit(x_mfcc)
    scaler_mfcc.transform(x_mfcc)
    x_mfcc = np.reshape(x_mfcc, [x_mfcc.shape[0], s1, s2])
    train_dataset = TensorDataset([torch.Tensor(x_text).float().to('cuda'), torch.Tensor(x_vid).float().to('cuda'),
                                   torch.Tensor(vid_seq).int().to('cuda'), torch.Tensor(x_mfcc).float().to('cuda'),
                                   torch.Tensor(x_pros).float().to('cuda'), torch.Tensor(aud_seq).int().to('cuda'),
                                   torch.Tensor(labels).long().to('cuda')])
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    params.n_train = len(x_text)
    # get dev data
    x_text, x_vid, vid_seq, x_mfcc, x_pros, aud_seq, labels = get_text_video_audio_data(params.data_path, 'dev')
    x_mfcc = np.reshape(x_mfcc, [x_mfcc.shape[0], -1])
    scaler_mfcc.transform(x_mfcc)
    x_mfcc = np.reshape(x_mfcc, [x_mfcc.shape[0], s1, s2])
    dev_dataset = TensorDataset([torch.Tensor(x_text).float().to('cuda'), torch.Tensor(x_vid).float().to('cuda'),
                                   torch.Tensor(vid_seq).int().to('cuda'), torch.Tensor(x_mfcc).float().to('cuda'),
                                   torch.Tensor(x_pros).float().to('cuda'), torch.Tensor(aud_seq).int().to('cuda'),
                                   torch.Tensor(labels).long().to('cuda')])
    dev_loader = DataLoader(dev_dataset, batch_size=params.batch_size, shuffle=False)
    params.n_dev = len(x_text)
    # get test data
    x_text, x_vid, vid_seq, x_mfcc, x_pros, aud_seq, labels = get_text_video_audio_data(params.data_path, 'test')
    x_mfcc = np.reshape(x_mfcc, [x_mfcc.shape[0], -1])
    scaler_mfcc.transform(x_mfcc)
    x_mfcc = np.reshape(x_mfcc, [x_mfcc.shape[0], s1, s2])
    test_dataset = TensorDataset([torch.Tensor(x_text).float().to('cuda'), torch.Tensor(x_vid).float().to('cuda'),
                                 torch.Tensor(vid_seq).int().to('cuda'), torch.Tensor(x_mfcc).float().to('cuda'),
                                 torch.Tensor(x_pros).float().to('cuda'), torch.Tensor(aud_seq).int().to('cuda'),
                                 torch.Tensor(labels).long().to('cuda')])
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)
    params.n_test = len(x_text)
    # train
    params.num_epochs = 20000 # give a random big number
    params.when = 10 # reduce LR patience
    params.txt_dim = 300
    params.vid_dim = 2048
    params.aud_dim = 120
    params.pros_dim = 35
    count = 0
    import sys
    test_loss = train_tva_1.initiate(params, train_loader, dev_loader, test_loader)    
