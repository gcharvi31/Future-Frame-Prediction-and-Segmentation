import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Adam
import torchvision.transforms as transforms
#from torchsummary import summary
from torch.utils.data import Dataset,DataLoader, random_split

from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
import albumentations as A



"""
Dataset class to load frames and their masks
"""
class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None,val=False):
        self.root_dir = root_dir
        self.transform = transform
        self.val=val     #set to true if loading videos from validation set

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if self.val:
            idx+=1000       # Videos 1000-19999 are validation videos
        video_path = os.path.join(self.root_dir, 'video_{}'.format(idx))
        frame=[]
        mask_list=[]
        for fn in range(22):
            image_path=os.path.join(video_path,'image_{}.png'.format(fn))
            img=np.array(Image.open(image_path))
            mask_path = os.path.join(video_path, 'mask.npy')
            masks = np.load(mask_path)
            mask=masks[fn]
            if self.transform is not None:
                aug = self.transform(image=img,mask=mask)
                img = aug['image']
                mask = aug['mask']
            frame.append(img)
            mask_list.append(mask)
        return frame,mask_list


"""
Encoding Block for Unet model
"""
class encoding_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(encoding_block,self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*model)
    def forward(self, x):
        return self.conv(x)

"""
Unet Model
"""
class unet_model(nn.Module):
    def __init__(self,out_channels=49,features=[64, 128, 256, 512]):
        super(unet_model,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1 = encoding_block(3,features[0])
        self.conv2 = encoding_block(features[0],features[1])
        self.conv3 = encoding_block(features[1],features[2])
        self.conv4 = encoding_block(features[2],features[3])
        self.conv5 = encoding_block(features[3]*2,features[3])
        self.conv6 = encoding_block(features[3],features[2])
        self.conv7 = encoding_block(features[2],features[1])
        self.conv8 = encoding_block(features[1],features[0])        
        self.tconv1 = nn.ConvTranspose2d(features[-1]*2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)        
        self.bottleneck = encoding_block(features[3],features[3]*2)
        self.final_layer = nn.Conv2d(features[0],out_channels,kernel_size=1)
    def forward(self,x):
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)        
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x


"""
Helper function to check accuracy of predicted mask and true mask
"""
def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x=torch.stack(x,dim=1)
            y=torch.stack(y,dim=1)
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            for i in range(22):
                d=x[:,i,:,:,:]
                t=y[:,i,:,:]
                softmax = nn.Softmax(dim=1)
                preds = torch.argmax(softmax(model(d)),axis=1)

                num_correct += (preds == t).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * t).sum()) / ((preds + t).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


# Path to root directory containing train,val,unlabelled folders
raw_dir="/vast/cg4177/Dataset_Student"  #CHANGE PATH 

t1 = A.Compose([A.Resize(160,240),
                A.augmentations.transforms.Normalize(mean=[0.5, 0.5,0.5], std=[0.5,0.5,0.5]),
                ToTensorV2()])

train_dataset = VideoDataset(root_dir=os.path.join(raw_dir,'train') ,transform=t1)
val_dataset= VideoDataset(root_dir=os.path.join(raw_dir,'val'),transform=t1,val=True)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,num_workers=1)
val_loader=DataLoader(val_dataset,batch_size=8,shuffle=True,num_workers=1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = unet_model().to(DEVICE)
LEARNING_RATE = 1e-4
num_epochs = 10        
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()
old_val_loss=float('inf')

"""
Training Loop
"""
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(enumerate(train_loader),total=len(train_loader))
    for batch_idx, (data, targets) in loop:
        data=torch.stack(data,dim=1)
        targets=torch.stack(targets,dim=1)
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)
        targets = targets.type(torch.long)
        #forward pass
        for i in range(22):
            d=data[:,i,:,:,:].to(DEVICE)
            t=targets[:,i,:,:].to(DEVICE)

            with torch.cuda.amp.autocast():
                predictions = model(d)
                loss = loss_fn(predictions, t)
            #backward prop
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #update tqdm loop
            loop.set_postfix(loss=loss.item())
    model.eval()
    val_loss=0.0
    with torch.no_grad():
        for (data,targets) in val_loader:
            data=torch.stack(data,dim=1)
            targets=torch.stack(targets,dim=1)
            data=data.to(DEVICE)
            targets=targets.to(DEVICE)
            targets = targets.type(torch.long)
            for i in range(22):
                d=data[:,i,:,:,:].to(DEVICE)
                t=targets[:,i,:,:].to(DEVICE)
                with torch.cuda.amp.autocast():
                    output=model(d)
                    loss=loss_fn(output,t)
                val_loss+=loss.item()
    if val_loss<old_val_loss:       #save model if validation loss decreases
        torch.save({
            'epoch':epoch,
            'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, 'segmentation_model.pth')
        old_val_loss = val_loss
    print(f"Epoch [{epoch + 1}/{num_epochs}] - val loss: {val_loss/len(val_loader):.4f}")


