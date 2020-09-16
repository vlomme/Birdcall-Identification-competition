import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset
import glob, os,time, random, librosa, argparse
from efficientnet_pytorch import EfficientNet
from hparams import Hparams, get_model, mono_to_color,random_power, accuracy, get_melspectr, test_accuracy
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from tqdm import tqdm
from scipy.stats.mstats import gmean
        
def preprocess():
    os.makedirs(hp.mel_folder, exist_ok=True)
    dataset = glob.glob(os.path.join("./short/", '**/*.wav'), recursive=True)
    with open('meta.csv', 'w', encoding="utf-8") as output:
        output.write("file,bird\n")
        for train_path in tqdm(dataset):
            # Only the name of the file | Только название файла
            wav_name = os.path.basename(train_path) 
            dirname = train_path.split("\\")[1]
            
            # Create melspectrogram | Получить спектр
            mel = get_melspectr(train_path)
            
            # Translate to torch | Перевести в torch
            mel = torch.from_numpy(mel)
            
            # Getting a new path to the file | Получаем новый путь до файла
            wav_path = os.path.join(hp.mel_folder, wav_name)

            # Save melspectrogram | Сохраняем иелспектрограмму
            save_path = wav_path.replace('.mp3', '.amp')
            torch.save(mel, save_path)
            output.write(wav_name +',' + dirname + "\n")
    secondary_labels = pd.read_csv('train1.csv')
    meta = pd.read_csv('meta.csv')
    meta = meta.merge(secondary_labels[["filename",'labels_bg']], how = 'left', left_on='file', right_on='filename',copy=False)
    meta[["file","bird",'labels_bg']].to_csv('meta.csv', index=False)
    print('Training dataset created / Тренировочный датасет создан')
    
class MelDataset(Dataset):
    def __init__(self, bird_list, hp):
        # Initialize the list of melspectrograms | Инициализировать список мелспектрограмм
        self.bird_list = bird_list
        self.hp = hp
        self.noise = pd.read_csv("nocall.csv")
        self.stop_border = 0.3 # Probability of stopping mixing | Вероятность прервать смешивание
        self.level_noise = 0.05 # level noise | Уровень шума
        self.div_coef = 100 # signal amplification during mixing | Усиления сигнала при смешивании
            
    def __len__(self):
        return len(self.bird_list)

    def __getitem__(self, idx):
        idx2 = random.randint(0, len(self.bird_list)-1) # Second file | Второй файл
        idx3 = random.randint(0, len(self.bird_list)-1) # Third file | Третий файл

        y = torch.zeros(self.hp.count_bird[0])
        birds, background = [],[]
        
        # Length of the segment | Длительность отрезка
        self.len_chack = random.randint(self.hp.len_chack[0]-48, self.hp.len_chack[0]+52)
        #self.len_chack = self.hp.len_chack[0]
        
        images = np.zeros((self.hp.n_mels, self.len_chack)).astype(np.float32)            
        for i,idy in enumerate([idx,idx2,idx3]):
            # Choosing a record with a bird | Выбираем запись с птицей
            sample = self.bird_list.loc[idy, :]
            # Uploading a record with a bird | Загружаем запись с птицей
            mel = torch.load(self.hp.mel_folder+sample.file.replace(".mp3",".amp")).numpy()

            # Birds in the file | Птицы в файле
            labels_bird = sample.bird.split()
            for bird in labels_bird:
                if not bird in birds and bird != 264:
                    birds.append(self.hp.BIRD_CODE[bird])
            
            # Birds in the background | Птицы на фоне     
            if sample.labels_bg:
                labels_bg = sample.labels_bg.split()
                for bg in labels_bg:
                    if not bg in background:
                        background.append(self.hp.BIRD_CODE[bg])
            
            # Select the piece that contains the sound | Выбираем кусок в котором содержится звук
            if mel.shape[1]>self.len_chack: 
                start = random.randint(0, mel.shape[1] - self.len_chack - 1)
                mel = mel[:, start : start + random.randint(self.len_chack-48, self.len_chack)]
            else:
                len_zero = random.randint(0, self.len_chack-mel.shape[1])
                mel = np.concatenate((np.zeros((self.hp.n_mels,len_zero)),mel), axis=1)
            
            mel = np.concatenate((mel,np.zeros((self.hp.n_mels,self.len_chack-mel.shape[1]))), axis=1)
            
            # Change the contrast | Изменить контрастность
            mel = random_power(mel, power = 3, c= 0.5)
            #mel = librosa.power_to_db(mel.astype(np.float32), ref=np.max)
            #mel = (mel+80)/80
            
            # Mix the signal | Смешать сигнал
            images = images + mel*(random.random() * self.div_coef + 1)
            
            # Abort accidentally | Случайно прервать 
            if random.random()<self.stop_border:
                break
        
        # Add a different sound without birds | Добавить другой звук без птиц
        idy = random.randint(0, len(self.noise)-1)
        sample = self.noise.loc[idy, :]
        mel = torch.load('./mel/'+sample.file.replace(".wav",".amp")).numpy()
        mel = np.concatenate((np.zeros((self.hp.n_mels,self.len_chack)),mel), axis=1)
        mel = np.concatenate((mel,np.zeros((self.hp.n_mels,self.len_chack))), axis=1)
        start = random.randint(0, mel.shape[1] - self.len_chack - 1)
        mel = mel[:, start : start + self.len_chack]
        mel = random_power(mel)
        #mel = librosa.power_to_db(mel.astype(np.float32), ref=np.max)
        #mel = (mel+80)/80
        images = images + mel/(mel.max()+0.0000001)*(random.random()*1+0.5)*images.max()
        
        # In db and normalize | В Дб и нормализовать
        images = librosa.power_to_db(images.astype(np.float32), ref=np.max)
        images = (images+80)/80
        
        # Add noise | Добавить шум
        # Add white noise | Добавить белый шум            
        if random.random()<0.9:
            images = images + (np.random.sample((self.hp.n_mels,self.len_chack)).astype(np.float32)+9) * images.mean() * self.level_noise * (np.random.sample() + 0.3)
        
        # Add pink noise | Добавить розовый шум
        if random.random()<0.9:
            r = random.randint(1,self.hp.n_mels)
            pink_noise = np.array([np.concatenate((1 - np.arange(r)/r,np.zeros(self.hp.n_mels-r)))]).T
            images = images + (np.random.sample((self.hp.n_mels,self.len_chack)).astype(np.float32)+9) * 2  * images.mean() * self.level_noise * (np.random.sample() + 0.3)
        
        # Add bandpass noise | Добавить полосовой шум
        if random.random()<0.9:
            a = random.randint(0, self.hp.n_mels//2)
            b = random.randint(a+20, self.hp.n_mels)
            images[a:b,:] = images[a:b,:] + (np.random.sample((b-a,self.len_chack)).astype(np.float32)+9) * 0.05 * images.mean() * self.level_noise  * (np.random.sample() + 0.3)
        
        
        # Lower the upper frequencies | Понизить верхние частоты
        if random.random()<0.5:
            images = images - images.min()
            r = random.randint(self.hp.n_mels//2,self.hp.n_mels)
            x = random.random()/2
            pink_noise = np.array([np.concatenate((1-np.arange(r)*x/r,np.zeros(self.hp.n_mels-r)-x+1))]).T
            images = images*pink_noise
            images = images/images.max()
        
        # Change the contrast | Изменить контрастность
        images = random_power(images, power = 2, c= 0.7)
        
        # Expand to 3 channels | Расширить до 3 каналов
        #images = torch.from_numpy(np.stack([images, images, images])).float()
        images = mono_to_color(images,hp.len_chack[0])

        # Draw pictures | Рисуем графики
        if random.random()<0.0001:
            img = images.numpy()
            img = img - img.min()
            img = img/img.max()
            img = np.moveaxis(img, 0, 2)
            imgplot = plt.imshow(img)
            plt.savefig('log/img/'+("_".join(self.hp.INV_BIRD_CODE[x] for x in birds))+'_'+sample.file+'.png')    
        
        # If there are no birds, then the background | Усли нет птиц, значит фон
        if not birds:
            birds.append(264)        
        
        # The background is 0.3, and the marked bird is 1 | Фон это 0.3, а помеченая птица 1
        for bird in background:
            if bird < len(y):
                y[bird]=0.3
        for bird in birds:
            #if not bird==264:
            y[bird]=1
        return images, y



def train(model,optimizer,epochs,train_accuracy,all_loss,best_bird_count,best_score, t_scores, f1_scores, b_scores):
    # Create a folder for logs | Создать папку для логов
    save_dir = os.path.join("./log")
    os.makedirs(save_dir, exist_ok=True)
    
    # Upload a list of training files | Загрузить список тренировочных mel meta.csv
    bird_list = pd.read_csv("meta_all.csv")
    bird_list = bird_list[bird_list.bird.isin(hp.bird_count)].reset_index(drop=True)
    bird_list = bird_list.fillna(0)
    train_count = len(bird_list)
    trainset = MelDataset(bird_list, hp)
    train_loader = data.DataLoader(trainset, batch_size = hp.batch_size, shuffle=True, drop_last=True, num_workers = 2)
    
    # Training process | Процесс обучения
    prediction_dict = {}
    start = time.time()
    model.zero_grad() 
    for epoch in range(epochs, 1000):
        step = 0
        model.train()
        start_time = time.time()
        for (mel, background) in train_loader:
            step+=1
            # Consider the network output | Считаем выход сети
            prediction = model(mel.cuda())
            
            # We consider an error | Считаем ошибку
            train_loss = nn.BCEWithLogitsLoss()(prediction, background.cuda())
            #train_loss = nn.CrossEntropyLoss()(prediction, np.argmax(background, axis = 1).cuda())
            
            # Calculate the gradients and make a step | Вычисляем градиенты и делаем шаг 
            train_loss.backward()
            if not step % (100//hp.batch_size): 
                optimizer.step()
                model.zero_grad()
            
            # Saving error and accuracy | Сохраняем ошибку и точность
            train_accuracy.append(accuracy(background, prediction))
            all_loss.append(train_loss.detach().cpu().numpy())
            
            # Every hp.save_interval steps we display statistics | Каждые 100 шагов выводим статистику
            if not step % hp.save_interval:
                print(str(epoch)+' '+str(step)+'/'+str(train_count//hp.batch_size), 
                        "время: %.3f loss: %.3f accuracy: %.3f " % (
                        (time.time()-start_time)/hp.save_interval,
                        np.mean(all_loss[-hp.save_interval:])*10,
                        np.mean(train_accuracy[-hp.save_interval:])))
                # Test | Тестируем 
                (bird_count, bird_accuracy, test_accuracy), _ = generate([model],epochs,hp.border,True)
                if bird_accuracy>0:
                    t_scores.append(bird_accuracy)
                    f1_scores.append(test_accuracy)
                    b_scores.append(bird_count)
                
                model.train()
                
                # Draw graphs | Рисуем графики
                plt.clf()
                plt.plot(gaussian_filter1d(train_accuracy[80:], 20))
                plt.plot(gaussian_filter1d(all_loss[80:], 20)*10)
                plt.savefig('log/all_loss.png')        
                plt.clf()
                plt.plot(t_scores)
                plt.savefig('log/t.png') 
                plt.clf()
                plt.plot(f1_scores)
                plt.savefig('log/f1.png')
                plt.clf()
                plt.plot(b_scores)
                plt.savefig('log/b.png')
                # Saving the model | Сохраняем модель
                if (bird_count>best_bird_count or test_accuracy>best_score or step==hp.save_interval):
                    if bird_count>best_bird_count:
                        best_bird_count = bird_count
                    if test_accuracy>best_score:
                        best_score = test_accuracy
                    
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_bird_count': bird_count,
                        'best_score': test_accuracy,
                        'train_accuracy': train_accuracy,
                        'all_loss': all_loss,
                        't_scores': t_scores,
                        'f1_scores': f1_scores,
                        'b_scores': b_scores,                        
                    }, 'log/enet_%d_%.3f_%.3f.pt' % (bird_count,bird_accuracy,test_accuracy))
                    print("Модель сохранена")
                start_time = time.time()


def generate(models, epochs, border,log_stat):
    start = time.time() 
    preds = []

    # Uploading a list of files for testing | Загружаем список файлов для тестирования
    TEST_FOLDER = f'{BASE_TEST_DIR}/test_audio/'
    test_info = pd.read_csv(f'{BASE_TEST_DIR}/test2.csv')
    
    # Looking for all unique audio recordings | Ищем все уникальные аудиозаписи
    unique_audio_id = test_info.audio_id.unique() 
    
    # Predict | Предсказываем
    for model in models:
        model.eval()
    with torch.no_grad():    
        for audio_id in unique_audio_id:
            # Getting a spectrogram | Получаем спектрограмму
            melspectr = get_melspectr(TEST_FOLDER + audio_id + ".mp3")
            melspectr = librosa.power_to_db(melspectr, amin=1e-7, ref=np.max)
            melspectr = ((melspectr+80)/80).astype(np.float16)
            
            # Looking for all the excerpts for this sound | Ищем все отрывки для данного звука  
            test_df_for_audio_id = test_info.query(f"audio_id == '{audio_id}'").reset_index(drop=True)
            est_bird =np.zeros((265))
            probass = {}
            
            # Проходим по все отрывкам 
            for index, row in test_df_for_audio_id.iterrows():
                # Getting the site, start time, and id | Получаем сайт, время начала и id
                site = row['site']
                start_time = row['seconds'] - 5
                row_id = row['row_id']
                mels = []
                probas = None
                
                # Cut out the desired piece | Вырезаем нужный кусок
                if site == 'site_1' or site == 'site_2':
                    start_index = int(hp.sr * start_time/hp.hop_length)
                    end_index = int(hp.sr * row['seconds']/hp.hop_length)                
                    y = melspectr[:,start_index:end_index]
                else:
                    y = melspectr
                    
                # cutting off the tail | отсекаю хвост
                if (y.shape[1]%hp.len_chack[0]):
                    y = y[:,:-(y.shape[1]%448)]
                
                prob = []
                for i,model in enumerate(models):
                    mels = []
                    probas = None                    
                    # Split into several chunks with the duration hp.len_chack | Разбиваем на несколько кусков длительностью hp.len_chack
                    ys = np.reshape(y, (hp.n_mels, -1, hp.len_chack[i]))
                    ys = np.moveaxis(ys, 1, 0)

                    # For each piece we make transformations | Для каждого куска делаем преобразования
                    for image in ys:
                        # Convert to 3 colors and normalize | Переводим в 3 цвета и нормализуем
                        image = image/image.max()
                        #image = image**0.85
                        #image = torch.from_numpy(np.stack([image, image, image])).float()
                        image = mono_to_color(image,hp.len_chack[i])
                        mels.append(image)

                    mels = np.stack(mels)                
                    
                    # Прохожу по всем batch
                    for n in range(0,len(mels),hp.batch_size):
                        if len(mels) == 1:
                            mel = np.array(mels)
                        else:
                            mel = mels[n:n+hp.batch_size]

                        mel = torch.from_numpy(mel).cuda()

                        # Predict | Получить выход модели
                        prediction = model(mel)
                        #prediction = F.softmax(prediction, dim=1)
                        prediction = torch.sigmoid(prediction)

                        # in numpy
                        proba = prediction.detach().cpu().numpy()

                        # Add zeros up to 265 | Добавить нули до 265
                        proba = np.concatenate((proba,np.zeros((proba.shape[0],265-proba.shape[1]))), axis=1)

                        # Adding to the array | Добавляю в массив
                        if not probas is None:
                            probas = np.append(probas, proba, axis = 0)
                        else:
                            probas = proba
                        if hp.len_chack[i] == 448:
                            probas = np.append(probas, proba, axis = 0)
                    prob.append(probas)

                # Averaging the ensemble | Усредняю ансамбль
                prob = np.stack(prob,axis=0)
                prob = prob**2
                proba = prob.mean(axis=0)#gmean(prob)/2 + prob.mean(axis=0)/2
                proba = proba**(1/2)
                
                # If a bird is encountered in one segment, increase its probability in others
                # Если встретилась птица в одном отрезке, увеличить её вероятность в других
                for xx in proba:
                    z = xx.copy()
                    z[z<0.5] = 0
                    est_bird = est_bird + z/70
                    est_bird[(est_bird<0.15)&(est_bird>0)] = 0.15
      
                # Dictionary with an array of all passages | Словарь с массивом всех отрывков
                probass[row_id] = proba
            
            est_bird[est_bird>0.3] = 0.3
            for row_id,probas in probass.items():
                prediction_dict = []
                for proba in probas:
                    proba += est_bird
                    events = proba > border
                    labels = np.argwhere(events).reshape(-1).tolist()

                    # To convert in the name of the bird | Преобразовать в название птиц
                    if len(labels) == 0  or (264 in labels):
                        continue
                    else:
                        labels_str_list = list(map(lambda x: hp.INV_BIRD_CODE[x], labels))
                        for i in labels_str_list:
                            if i not in prediction_dict:
                                prediction_dict.append(i)  
                    
                # If birds are not predicted | Если не предсказываются птицы
                if len(prediction_dict) == 0:
                    prediction_dict = "nocall"
                else:
                    prediction_dict = " ".join(prediction_dict)
          
                # To add to the list | Добавить в список
                preds.append([row_id, prediction_dict])

        # Convert to DataFrame and save | Перевести в DataFrame и сохранить
        preds = pd.DataFrame(preds, columns=['row_id', 'birds'])
        preds.to_csv('submission.csv', index=False)
    
    return test_accuracy(preds,log_stat,border), time.time() - start


def pseudo(models):
    files = {}
    with open('meta_all.csv', 'r', encoding="utf-8") as input:
        input.readline()
        for s in input:
            s = s.strip().split(',')
            file,bird,background = s[0],s[1],s[2].split(' ')
            file = file.split('.')[0]
            files[file] = [bird,background]
    
    # Uploading a list of files for marking | Загружаем список файлов для маркирования
    dataset = glob.glob(os.path.join("./mel/", 'XC*.amp'), recursive=True)
    #dataset = glob.glob(os.path.join("./un_bird/", '*.wav'), recursive=True)
    #dataset = ['./clear/osprey/XC27026.mp3']
    prediction_dict = {}
    
    # Predict | Предсказываем
    for model in models:
        model.eval()
    with torch.no_grad():
        for file_name in tqdm(dataset):
            #y = get_melspectr(file_name)
            y = torch.load(file_name).numpy()
            est_bird =np.zeros((265))
            mels = []
            probas = None
            ys = []

            if y.shape[1]>=hp.len_chack:
                for i in range(0,(y.shape[1]*5)//hp.len_chack-4,3):
                    yy = y[:,i*hp.len_chack//5:(i+5)*hp.len_chack//5]
                    if yy.shape[1]<hp.len_chack:
                        yy = np.concatenate((yy,np.zeros((hp.n_mels,hp.len_chack-yy.shape[1]))), axis=1)
                    ys.append(yy)
                ys = np.stack(ys) 
            else:
                yy = np.concatenate((y,np.zeros((hp.n_mels,hp.len_chack-y.shape[1]))), axis=1)
                ys = np.array([yy])
            
            for y in ys:
                y = librosa.power_to_db(y,  amin=1e-7, ref=np.max)
                y = ((y+80)/80).astype(np.float16)
                image = torch.from_numpy(np.stack([y, y, y])).float()
                mels.append(image)

            mels = np.stack(mels) 
            for n in range(0,len(mels),hp.batch_size*2):
                if len(mels) == 1:
                    mel = np.array(mels)
                else:
                    mel = mels[n:n+hp.batch_size*2]

                mel = torch.from_numpy(mel).cuda()
                
                prob = []
                for model in models:
                    prediction = model(mel)
                    prediction = torch.sigmoid(prediction)
                    proba = prediction.detach().cpu().numpy()
                    proba = np.concatenate((proba,np.zeros((proba.shape[0],265-proba.shape[1]))), axis=1)
                    prob.append(proba)
                    
                prob = np.stack(prob)
                proba = prob.mean(axis=0)
                            
                if not probas is None:
                    probas = np.append(probas, proba, axis = 0)
                else:
                    probas = proba
            
            for sk,proba in enumerate(probas):
                proba += est_bird
                file_name1 = os.path.basename(file_name).replace('.mp3','').replace('.wav','').replace('.amp','')
                sek = file_name1 +'_' + str(sk)+'.amp'
                events = proba > hp.border
                labels = np.argwhere(events).reshape(-1).tolist()
                if len(labels) == 0  or (264 in labels):
                    continue
                else:
                    labels_str_list = list(map(lambda x: hp.INV_BIRD_CODE[x], labels))
                    if file_name1 in files:
                        bird = []
                        background = []
                        for i in labels_str_list:
                            if i in files[file_name1][1] or i==files[file_name1][0]:
                                if not i in bird:
                                    bird.append(i)
                            else:
                                if not i in background:
                                    background.append(i)
                        if bird:
                            torch.save(torch.from_numpy(ys[sk].astype(np.float16)), './mel_p/'+sek)
                            prediction_dict[sek] = [" ".join(bird)," ".join(background)]
                            """
                            img = np.moveaxis(mels[sk], 0, 2)
                            #ys[sk] = ys[sk]/ys[sk].max()
                            #img = np.moveaxis(np.stack([ys[sk],ys[sk],ys[sk]]), 0, 2)
                            img = np.array(img, dtype = np.float64)
                            plt.imshow(img)
                            plt.savefig('./1/'+sek+" ".join(bird)+'.png') 
                            """                            

        preds = pd.DataFrame.from_dict(prediction_dict, orient='index', columns=['birds',"labels_bg"])
        preds.reset_index(inplace=True)
        preds.columns = ['file', 'bird',"labels_bg"]
        preds.to_csv('meta.csv', index=False)


# Loading hp | Загружаем гиперпараметры
hp = Hparams()

if __name__ == "__main__":
    BASE_TEST_DIR = '.'
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", default='train', help=\
        "Enter the function you want to run | Введите функцию, которую надо запустить (preprocess, train, generate)")
    args = parser.parse_args()
    
    if args.run == 'preprocess' or args.run == 'p':
        preprocess()    
    else:
        # to create a model | создать model
        all_model = []
        for i in range(len(hp.models_name)):
            model,optimizer, epochs, train_accuracy, all_loss, best_bird_count, best_score, t_scores, f1_scores, b_scores = get_model(hp.models_name[i],hp.chk[i],hp.count_bird[i])
            all_model.append(model)
        if args.run == 'train' or args.run == 't':
            train(all_model[0],optimizer,epochs,train_accuracy,all_loss,best_bird_count,best_score, t_scores, f1_scores, b_scores)
        elif args.run == 'pseudotarget' or args.run == 'm':
            pseudo(all_model)
        elif args.run == 'generate' or args.run == 'g':
            for i in [ 0.4, 0.5, 0.6]:
                print(i, generate(all_model, epochs, i, True))
        else:
            print("Enter the correct function | Введите корректную функцию (preprocess, train, generate)")  