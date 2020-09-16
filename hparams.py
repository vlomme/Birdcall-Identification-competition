import numpy as np
import cv2, librosa, random, torch
import pandas as pd
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from sklearn.metrics import f1_score
from torch.optim import Adam
import torch.nn.functional as F


class Hparams():
    def __init__(self):
        #resnet50 resnext50_32x4d mobilenet_v2 efficientnet-b3  densenet121 densenet169 
        self.models_name = ['resnet50','efficientnet-b0','efficientnet-b0','efficientnet-b0','efficientnet-b0','resnet50']
        #self.chk = ['resnet50_78_0.830_0.666.pt','enet0_101_0.771_0.692.pt','enet0_45_0.558.pt','enet0_133_0.707_0.691.pt',
        #            '150enet0_116_0.707_0.703.pt','2.5resnet50_113_0.715_0.693.pt']
        self.chk = ['']
        self.count_bird = [265,265,265,265,150,265] #count birds|Количество птиц, 264 - all, 265 + nocall
        self.len_chack = [448,448,448,448,448,224] # The duration of the training files 448 = 5 second|Длительность обучающих файлов
        
        self.mel_folder = './mel/'
        self.n_fft = 892
        self.sr = 21952 
        self.hop_length=245
        self.n_mels =  224
        self.win_length = self.n_fft
        self.batch_size = 10 # 3 - b7, 8 - b5,  12 - b3, 25 - b0, 18 - b1 70
        self.lr = 0.001
        self.border = 0.5
        self.save_interval = 20 #Model saving interval
        # Список из count_bird птиц по пополуярности
        self.bird_count = pd.read_csv('bird_count.csv').ebird_code.to_numpy()        
        self.BIRD_CODE = {b:i for i,b in enumerate(self.bird_count)}
        self.INV_BIRD_CODE = {v: k for k, v in self.BIRD_CODE.items()}
        self.bird_count = self.bird_count[:self.count_bird[0]]


hp = Hparams()
def mono_to_color(X: np.ndarray,len_chack, mean=0.5, std=0.5, eps=1e-6):
    trans = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize([hp.n_mels, len_chack]), transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    X = np.stack([X, X, X], axis=-1)
    V = (255 * X).astype(np.uint8)
    V = (trans(V)+1)/2
    return V
    
    
def accuracy(y_true, y_pred):
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.detach().cpu().numpy()
    return f1_score(y_true > hp.border, y_pred > hp.border, average="samples")
    
    
def get_melspectr(train_path):
    # Load file | Загружаем файл
    y, _ = librosa.load(train_path,sr=hp.sr,mono=True,res_type="kaiser_fast")

    # Create melspectrogram | Создать Мелспектрограмму
    spectr = librosa.feature.melspectrogram(y, sr=hp.sr, n_mels=hp.n_mels, n_fft=hp.n_fft, hop_length = hp.hop_length, win_length = hp.win_length, fmin = 300)
    return spectr.astype(np.float16)


def random_power(images, power = 1.5, c= 0.7):
    images = images - images.min()
    images = images/(images.max()+0.0000001)
    images = images**(random.random()*power + c)
    return images

    
def test_accuracy(preds, log_stat= False, border=0.5):
    answer = pd.read_csv('example_test_audio_summary.csv')
    preds = answer.merge(preds, how = 'right', left_on='filename_seconds', right_on='row_id')
    y_true, y_pred = [], []
    my_bird = 0
    pred_bird = 0
    bad_bird = {}    
    for all in preds.loc[:,['bird','birds']].to_numpy(): 
        y = np.zeros(265)
        c = np.array(all[0].split())
        for bird in c:
            y[hp.BIRD_CODE[bird]]=1
        y_true.append(y)
        
        y = np.zeros(265)
        d = np.array(all[1].split())
        for bird in d:
            y[hp.BIRD_CODE[bird]]=1
        y_pred.append(y)
        
        mask = np.in1d(d, c)
        #good += mask.sum()
        if d[0] != 'nocall':
            pred_bird += len(d)
        if mask.sum()>0 and d[0] != 'nocall':
            my_bird += mask.sum()
        for i in d[~mask]:
            if i in bad_bird:
                bad_bird[i] += 1
            else:
                bad_bird[i] = 1
        #all_bird += (len(c)+len(d))/2
    if not pred_bird: pred_bird = 1
    f1 = f1_score(y_true, y_pred, average="samples")
    print("border: %.1f bird: %d bird_accuracy: %.3f test_accuracy: %.3f" % (
                                border,my_bird, my_bird/pred_bird, f1)) 
    if log_stat:
        for w in sorted(bad_bird, key=bad_bird.get, reverse=True)[:5]:
            print (w, bad_bird[w])            
    
    return my_bird, my_bird/pred_bird, f1


class BirdcallNet( nn.Module):
    def __init__(self, name, num_classes=265):
        super(BirdcallNet, self).__init__()
        self.model = models.__getattribute__(name)(pretrained=True)
        if name in ["resnet50","resnext50_32x4d"]:
            self.model.fc = nn.Linear(2048, num_classes)
        elif name in ['resnet18','resnet34']:
            self.model.fc = nn.Linear(512, num_classes)
        elif  name =="densenet121":
            self.model.classifier = nn.Linear(1024, num_classes)
        elif name in ['alexnet','vgg16']:
            self.model.classifier[-1] = nn.Linear(4096, num_classes)
        elif name =="mobilenet_v2":
            self.model.classifier[1] = nn.Linear(1280, num_classes)
        #print(self.model)
    def forward(self, x):
        return self.model(x)

        
def get_model(model_name,chk,count_bird):
    best_bird_count,best_score, epochs = 0,0,1
    all_loss, train_accuracy = [], []
    f1_scores,t_scores,b_scores = [],[],[]
    if not chk and model_name in ['efficientnet-b3','efficientnet-b0']:
        model = EfficientNet.from_pretrained(model_name, num_classes = count_bird).cuda()
        optimizer = Adam(model.parameters(), lr = hp.lr)
    else:
        models_names = ['alexnet','resnet50','resnet18','resnet34','mobilenet_v2','densenet121','resnext50_32x4d','densenet169']
        if model_name in models_names:
            model = BirdcallNet(model_name, hp.count_bird[0]).cuda()
        elif model_name == 'mini':
            model = Classifier(hp.count_bird[0]).cuda()
        else:
            model = EfficientNet.from_name(model_name, override_params={'num_classes': count_bird }).cuda()
        optimizer = Adam(model.parameters(), lr = hp.lr)
        # Load a checkpoint | Загрузить чекпоинт
        if chk:
            ckpt = torch.load('log/'+chk)
            model.load_state_dict(ckpt['model'])
            epochs = int(ckpt['epoch']) + 1
            train_accuracy =  ckpt['train_accuracy'] 
            all_loss   = ckpt['all_loss'] 
            best_bird_count =  ckpt['best_bird_count'] 
            best_score   = ckpt['best_score']
            
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 't_scores' in ckpt:
                t_scores   = ckpt['t_scores']
            if 'f1_scores' in ckpt:
                f1_scores   = ckpt['f1_scores']
            if 'b_scores' in ckpt:
                b_scores   = ckpt['b_scores']
            print('Чекпоинт загружен: Эпоха %d Число обнаруженых птиц %d Score %.3f' % (epochs,best_bird_count,best_score))
    return model,optimizer, epochs, train_accuracy, all_loss, best_bird_count, best_score, t_scores, f1_scores, b_scores
    