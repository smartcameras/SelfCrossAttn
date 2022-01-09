import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

def get_text_video_audio_data(data_path, part='train'):
    if part == 'train':
        x_txt = np.load(data_path+'/'+'train_text.npy')
        x_vid = np.load(data_path+'/'+'train_video.npy')
        vid_seqN = np.load(data_path+'/'+'train_video_seqN.npy')
        x_mfcc = np.load(data_path+'/'+'train_audio_mfcc.npy')
        x_pros = np.load(data_path + '/' + 'train_audio_prosody.npy')
        aud_seqN = np.load(data_path + '/' + 'train_audio_seqN.npy')
        labels = np.load(data_path+'/'+'train_label.npy')
        if np.where(vid_seqN == 0)[0].any():
            tr_inds = np.where(vid_seqN == 0)
            vid_seqN = np.delete(vid_seqN, tr_inds, 0)
            x_vid = np.delete(x_vid, tr_inds, 0)
            x_txt = np.delete(x_txt, tr_inds, 0)
            x_mfcc = np.delete(x_mfcc, tr_inds, 0)
            x_pros = np.delete(x_pros, tr_inds, 0)
            aud_seqN = np.delete(aud_seqN, tr_inds, 0)
            labels = np.delete(labels, tr_inds, 0)
    elif part == 'dev':
        x_txt = np.load(data_path + '/' + 'dev_text.npy')
        x_vid = np.load(data_path+'/'+'dev_video.npy')
        vid_seqN = np.load(data_path+'/'+'dev_video_seqN.npy')
        x_mfcc = np.load(data_path + '/' + 'dev_audio_mfcc.npy')
        x_pros = np.load(data_path + '/' + 'dev_audio_prosody.npy')
        aud_seqN = np.load(data_path + '/' + 'dev_audio_seqN.npy')
        labels = np.load(data_path+'/'+'dev_label.npy')
        if np.where(vid_seqN == 0)[0].any():
            inds = np.where(vid_seqN == 0)
            vid_seqN = np.delete(vid_seqN, inds)
            x_vid = np.delete(x_vid, inds, 0)
            x_txt = np.delete(x_txt, inds, 0)
            x_mfcc = np.delete(x_mfcc, inds, 0)
            x_pros = np.delete(x_pros, inds, 0)
            aud_seqN = np.delete(aud_seqN, inds, 0)
            labels = np.delete(labels, inds)
    elif part == 'test':
        x_txt = np.load(data_path + '/' + 'test_text.npy')
        x_vid = np.load(data_path+'/'+'test_video.npy')
        vid_seqN = np.load(data_path+'/'+'test_video_seqN.npy')
        x_mfcc = np.load(data_path + '/' + 'test_audio_mfcc.npy')
        x_pros = np.load(data_path + '/' + 'test_audio_prosody.npy')
        aud_seqN = np.load(data_path + '/' + 'test_audio_seqN.npy')
        labels = np.load(data_path+'/'+'test_label.npy')
        if np.where(vid_seqN == 0)[0].any():
            inds = np.where(vid_seqN == 0)
            vid_seqN = np.delete(vid_seqN, inds)
            x_vid = np.delete(x_vid, inds, 0)
            x_txt = np.delete(x_txt, inds, 0)
            x_mfcc = np.delete(x_mfcc, inds, 0)
            x_pros = np.delete(x_pros, inds, 0)
            aud_seqN = np.delete(aud_seqN, inds, 0)
            labels = np.delete(labels, inds)
    else:
        x_txt = []
        x_vid = []
        vid_seqN = []
        x_mfcc = []
        x_pros = []
        aud_seqN = []
        labels = []
    return x_txt, x_vid, vid_seqN, x_mfcc, x_pros, aud_seqN, labels

def save_model(model, name):
    torch.save(model, name)

def load_model(name):
    model = torch.load(name)
    return model

# taken from https://github.com/david-yoon/attentive-modality-hopping-for-SER
'''
list_y_ture : reference (label)
list_y_pred : predicted value
note        : do not consider "label imbalance"
'''


def unweighted_accuracy(list_y_true, list_y_pred):
    assert (len(list_y_true) == len(list_y_pred))

    y_true = np.array(list_y_true)
    y_pred = np.array(list_y_pred)

    return accuracy_score(y_true=y_true, y_pred=y_pred)


'''
list_y_ture : reference (label)
list_y_pred : predicted value
note        : compute accuracy for each class; then, average the computed accurcies
              consider "label imbalance"
'''


def weighted_accuracy(list_y_true, list_y_pred):
    assert (len(list_y_true) == len(list_y_pred))

    y_true = np.array(list_y_true)
    y_pred = np.array(list_y_pred)

    w = np.ones(y_true.shape[0])
    for idx, i in enumerate(np.bincount(y_true)):
        w[y_true == idx] = float(1 / i)

    return accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=w)

def weighted_precision(list_y_true, list_y_pred):
    wa = precision_score(y_true=list_y_true, y_pred=list_y_pred, average='weighted')
    return wa

def unweighted_precision(list_y_true, list_y_pred):
    uwa = precision_score(y_true=list_y_true, y_pred=list_y_pred, average='macro')
    return uwa
