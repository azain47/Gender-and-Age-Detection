import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 5))

class UTKFaceDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.path = dataset_path
        self.dataframe = pd.read_csv('UTKFace2.csv')
        self.dataframe['age'] = scaler.fit_transform(self.dataframe['age'].values.reshape(-1,1))
        self.transform = transform

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
    
        image = Image.open(self.dataframe['filepath'][idx])

        if self.transform:
            image = self.transform(image)

        age = self.dataframe['age'][idx]
        gender = self.dataframe['gender'][idx]       
        
        sample = {
            'image': image,
            'labels': {
                'age': age,
                'gender': gender
            }
        }
        return sample
    
class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
         
    def __call__(self, tensor):
        return tensor + torch.randn(size=tensor.size()) * self.std + self.mean
     
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    AddGaussianNoise(0,0.07),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))   
])

dataset = UTKFaceDataset(dataset_path='./UTKFace/', transform=transform)

split = int(len(dataset)*0.75)

trainset, testset = torch.utils.data.random_split(dataset, [split, len(dataset)-split])

train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
test_loader = DataLoader(testset, batch_size=32, shuffle=True)

class CNN(nn.Module):
    def __init__(self, genders, ages) -> None:
        super(CNN,self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,256,3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(6400,1024)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256,128)
        self.gender = nn.Linear(128,genders)
        self.age = nn.Linear(128,ages)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.max_pool2d(self.conv4(x),kernel_size = 2)
        x = x.view(x.size(0),-1)
        x = F.dropout(x,0.2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,0.1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return {
            'age':F.relu(self.age(x)),
            'gender':self.gender(x)
        }
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CNN(2,1).to(device=device)
criterion_gender = nn.CrossEntropyLoss()
criterion_age = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 30

train_losses = np.zeros(epochs)
test_losses = np.zeros(epochs)

def train():
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        t0 = datetime.now()
        train_loss= []

        for images in (train_loader):

            model.train()
            data = images['image'].to(device)
            optimizer.zero_grad()
            
            outputs = model(data)
          
            age_loss = criterion_age((outputs['age'].view(outputs['age'].size(0))).float(), 
                                     images['labels']['age'].to(device).float())
            
            gender_losses = criterion_gender(outputs['gender'], 
                                             images['labels']['gender'].to(device))
            loss = age_loss + gender_losses

            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())
        
        train_loss = torch.tensor(train_loss).mean().item()

        test_loss = []

        for images in (test_loader):

            model.train()
            data = images['image'].to(device)
              
            outputs = model(data)

            age_loss = criterion_age(outputs['age'].view(outputs['age'].size(0)), images['labels']['age'].to(device).float())
            gender_losses = criterion_gender(outputs['gender'], images['labels']['gender'].to(device))
            loss = age_loss + gender_losses
            
            # print(f'TESTING: Gender Loss: {gender_losses.item()}, Age Loss: {age_loss.item()}')

            test_loss.append(loss.item())
        
        test_loss = torch.tensor(test_loss).mean().item()

        train_losses[epoch] = train_loss
        test_losses[epoch] = test_loss

        dt = datetime.now()-t0

        print(f'Epoch:{epoch+1}, Time taken:{dt}, Train loss:{train_loss:.4f},Test Loss:{test_loss:.4f}')
               
    plt.plot(train_losses,label='Train')
    plt.plot(test_losses,label='Test')
    plt.xlabel('#epochs')
    plt.ylabel('Loss')
    plt.title('Loss Graph')
    plt.show()
    torch.save(model,'model.pt')

# train()

model = torch.load('model.pt',map_location=device)

def calc_accuracy():
    model.eval()
    maeerrors = []
    mseerrors = []
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    with torch.no_grad():
        for data in test_loader:
            images = data['image'].to(device)
            labels = data['labels']['age'] 
            images,labels = images.to(device),labels.to(device)

            outputs = model(images)

            predicted = outputs['age'].view(outputs['age'].size(0))

            maeerrors.append(mean_absolute_error(predicted.cpu().numpy(), labels.cpu().numpy()))
            mseerrors.append(mean_squared_error(predicted.cpu().numpy(), labels.cpu().numpy()))

    maeerrors = np.array(maeerrors).mean()
    mseerrors = np.array(mseerrors).mean()
    print("Age errors:")
    print(f'Mean Absolute Error: {maeerrors}')
    print(f'Mean Sqaured Error: {mseerrors}')

    correct = 0
    total = 0

    for data in test_loader:
        images = data['image'].to(device)
        labels = data['labels']['gender'] 
        images,labels = images.to(device),labels.to(device)

        outputs = model(images)
            
        _,predicted = torch.max(outputs['gender'],1)

        correct+= (predicted == labels).sum().item()
        total += labels.shape[0]

    test_accuracy = correct/total
    
    print(f'Gender Accuracy: {test_accuracy*100}')

# Testing on Real People(ME)
import math
import argparse
import cv2 
import cvlib as cv

parser = argparse.ArgumentParser()
parser.add_argument('path_to_file')
parser.add_argument('-acc','--accuracy',action='store_true')
args = parser.parse_args()

if(args.accuracy):
    calc_accuracy()

tf=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((200,200)),
    transforms.ToTensor()
])

genderclass = ['Male', 'Female']

while args.path_to_file != '':
    frame = cv2.imread(args.path_to_file)
    x,y,z = frame.shape
    if x>1000 or y>1000:
        x *= 0.3
        y *= 0.3
        x =int(x)
        y= int(y)
    face, confidence = cv.detect_face(frame)

    for f in face:
        if x>1000 or y>1000:
            (startX, startY) = f[0]-100, f[1]-100
            (endX, endY) = f[2]+100, f[3]+100
        else:
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        face_crop = np.copy(frame[startY:endY , startX:endX])

        img = np.array(face_crop)

        test = tf(img)

        test = torch.reshape(test,(1,3,200,200))

        test = test.to(device)
        outputs =  model(test)
        _,gender = torch.max(outputs['gender'],1)

        with torch.no_grad():
            age = scaler.inverse_transform(outputs['age'].cpu().numpy())
        
        label = (f'Gender :{genderclass[gender.item()]}, Age : {math.ceil(age.item())}')
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        cv2.putText(frame,label,(startX,Y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
        print(label)
        while True:
            frame = cv2.resize(frame,(y,x))
            cv2.imshow('gender & age detect', frame)
            
            # cv2.resizeWindow('gender & age detect', 600, 800)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
