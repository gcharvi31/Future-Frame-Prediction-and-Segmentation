#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchmetrics



import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import torchmetrics
from torch.nn.parallel import DistributedDataParallel as DDP

import json
import math
import logging


import torchvision.utils as vutils
from torchvision import transforms


# In[2]:


class STConvLSTMCell(nn.Module):
    """
    Spatio-Temporal Convolutional LSTM Cell Implementation.
    """

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, forget_bias=1.0, layer_norm=True):
        super(STConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.forget_bias = forget_bias
        self.layer_norm = layer_norm

        self.conv_wx = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=7 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_wht_1 = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_wml_1 = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=3 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_wml = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_wcl = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_h = nn.Conv2d(
            in_channels=self.hidden_dim + self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=(1, 1),
            padding=0,
            bias=self.bias,
        )

        nn.init.orthogonal_(self.conv_wx.weight)
        nn.init.orthogonal_(self.conv_wht_1.weight)
        nn.init.orthogonal_(self.conv_wml_1.weight)
        nn.init.orthogonal_(self.conv_wml.weight)
        nn.init.orthogonal_(self.conv_wcl.weight)
        nn.init.orthogonal_(self.conv_h.weight)

        if self.layer_norm:
            self.conv_wx_norm = nn.BatchNorm2d(7 * self.hidden_dim)
            self.conv_wht_1_norm = nn.BatchNorm2d(4 * self.hidden_dim)
            self.conv_wml_1_norm = nn.BatchNorm2d(3 * self.hidden_dim)
            self.conv_wml_norm = nn.BatchNorm2d(self.hidden_dim)
            self.conv_wcl_norm = nn.BatchNorm2d(self.hidden_dim)
            self.conv_h_norm = nn.BatchNorm2d(self.hidden_dim)

        self.forget_bias_h = torch.nn.Parameter(torch.tensor(self.forget_bias))
        self.forget_bias_m = torch.nn.Parameter(torch.tensor(self.forget_bias))

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur, m_cur = cur_state

        conved_wx = self.conv_wx(input_tensor)
        conved_wht_1 = self.conv_wht_1(h_cur)
        conved_wml_1 = self.conv_wml_1(m_cur)

        if self.layer_norm:
            conved_wx = self.conv_wx_norm(conved_wx)
            conved_wht_1 = self.conv_wht_1_norm(conved_wht_1)
            conved_wml_1 = self.conv_wml_1_norm(conved_wml_1)

        wxg, wxi, wxf, wxg_, wxi_, wxf_, wxo = torch.split(conved_wx, self.hidden_dim, dim=1)
        whg, whi, whf, who = torch.split(conved_wht_1, self.hidden_dim, dim=1)
        wmg, wmi, wmf = torch.split(conved_wml_1, self.hidden_dim, dim=1)

        g_t = torch.tanh(wxg + whg)
        i_t = torch.sigmoid(wxi + whi)
        f_t = torch.sigmoid(wxf + whf + self.forget_bias_h)
        c_next = f_t * c_cur + i_t * g_t

        g_t_ = torch.tanh(wxg_ + wmg)
        i_t_ = torch.sigmoid(wxi_ + wmi)
        f_t_ = torch.sigmoid(wxf_ + wmf + self.forget_bias_m)
        m_next = f_t_ * m_cur + i_t_ * g_t_

        wco = self.conv_wcl(c_next)
        wmo = self.conv_wml(m_next)

        if self.layer_norm:
            wco = self.conv_wcl_norm(wco)
            wmo = self.conv_wml_norm(wmo)

        o_t = torch.sigmoid(wxo + who + wco + wmo)

        combined_cmn = torch.cat([c_next, m_next], dim=1)
        h_next = o_t * torch.tanh(self.conv_h(combined_cmn))

        return h_next, c_next, m_next


# In[3]:


class Generator(nn.Module):
    """
    Generator model with Spatio-Temporal Convolutional LSTMs.
    """

    def __init__(self, cfg, device):
        super(Generator, self).__init__()

        self.input_size = cfg["input_size"]
        self.hidden_dim = cfg["hidden_dim"]
        self.input_dim = cfg["input_dim"]
        self.kernel_size = tuple(cfg["kernel_size"])

        self.height, self.width = self.input_size
        self.device = device

        self.STConvLSTM_Cell_1 = STConvLSTMCell(
            input_size=self.input_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            bias=True,
        )

        self.STConvLSTM_Cell_2 = STConvLSTMCell(
            input_size=self.input_size,
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            bias=True,
        )

        self.STConvLSTM_Cell_3 = STConvLSTMCell(
            input_size=self.input_size,
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            bias=True,
        )

        self.STConvLSTM_Cell_4 = STConvLSTMCell(
            input_size=self.input_size,
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            bias=True,
        )

        self.head = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.input_dim, kernel_size=(1, 1))

    def forward(self, input_sequence, future=11):
        batch_size = input_sequence.size(0)

        hidden_initializer = [torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(self.device)] * 3

        h_t1, c_t1, m_t1 = hidden_initializer.copy()
        h_t2, c_t2, _ = hidden_initializer.copy()
        h_t3, c_t3, _ = hidden_initializer.copy()
        h_t4, c_t4, _ = hidden_initializer.copy()

        outputs = []
        seq_len = input_sequence.size(1)

        for time in range(seq_len):
            if time:
                m_t1 = m_t4

            h_t1, c_t1, m_t1 = self.STConvLSTM_Cell_1(
                input_tensor=input_sequence[:, time, :, :, :], cur_state=[h_t1, c_t1, m_t1]
            )
            h_t2, c_t2, m_t2 = self.STConvLSTM_Cell_2(input_tensor=h_t1, cur_state=[h_t2, c_t2, m_t1])
            h_t3, c_t3, m_t3 = self.STConvLSTM_Cell_3(input_tensor=h_t2, cur_state=[h_t3, c_t3, m_t2])
            h_t4, c_t4, m_t4 = self.STConvLSTM_Cell_4(input_tensor=h_t3, cur_state=[h_t4, c_t4, m_t3])

            output = self.head(h_t4)
            output = torch.sigmoid(output)
            outputs += [output]

        for t in range(future):
            m_t1 = m_t4

            h_t1, c_t1, m_t1 = self.STConvLSTM_Cell_1(input_tensor=outputs[-1], cur_state=[h_t1, c_t1, m_t1])
            h_t2, c_t2, m_t2 = self.STConvLSTM_Cell_2(input_tensor=h_t1, cur_state=[h_t2, c_t2, m_t1])
            h_t3, c_t3, m_t3 = self.STConvLSTM_Cell_3(input_tensor=h_t2, cur_state=[h_t3, c_t3, m_t2])
            h_t4, c_t4, m_t4 = self.STConvLSTM_Cell_4(input_tensor=h_t3, cur_state=[h_t4, c_t4, m_t3])

            output = self.head(h_t4)
            output = torch.sigmoid(output)
            outputs += [output]

        outputs = torch.stack(outputs, 1)

        return outputs


class Discriminator(nn.Module):
    """
    Discriminator model.
    """

    def __init__(self, cfg):
        super(Discriminator, self).__init__()

        self.input_size = cfg["input_size"]
        self.hidden_dim = cfg["hidden_dim"]
        self.height, self.width = self.input_size

        self.linear_1 = nn.Linear(self.height * self.width, self.hidden_dim * 4)
        self.linear_2 = nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2)
        self.linear_3 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear_4 = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.height * self.width)
        x = self.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.relu(self.linear_2(x))
        x = self.dropout(x)
        x = self.relu(self.linear_3(x))
        x = self.dropout(x)
        out = self.sigmoid(self.linear_4(x))

        return out


# In[4]:


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


# In[5]:


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


# In[6]:



class Config:
    """
    Config Parser class.
    """

    def __init__(self, file_path_or_dict, logger_name="global"):
        super(Config, self).__init__()

        self.logger = logging.getLogger(logger_name)
        self.cfg_init = self.load_config(file_path_or_dict)
        self.check_meta()

    def load_config(self, file_path_or_dict):
        if type(file_path_or_dict) is str:
            assert os.path.exists(file_path_or_dict), '"{}" not exists'.format(file_path_or_dict)
            config = dict(json.load(open(file_path_or_dict)))
        elif type(file_path_or_dict) is dict:
            config = file_path_or_dict
        else:
            raise Exception("The input must be a string path or a dict")

        return config

    def check_meta(self):
        if "meta" not in self.cfg_init:
            self.logger.warning("The cfg does not include meta tag, will generate default meta tag")
            self.logger.warning("Used the default meta configs.")

            self.cfg_init["meta"] = (
                {
                    "board_path": "board",
                },
            )
        else:
            cfg_meta = self.cfg_init["meta"]

            if "board_path" not in cfg_meta:
                self.logger.warning("Not specified board_path, used default. (board)")
                self.cfg_init["meta"]["board_path"] = "board"

        self.__dict__.update(self.cfg_init)

    def log_dict(self):
        self.logger.debug("Used config: \n {}".format(self.cfg_init))


# In[7]:


device="cuda" if torch.cuda.is_available() else "cpu"

cfg = Config('frame_pred/src/config_hpc.json')  #CHANGE PATH

lr = cfg.train["lr"]
epochs = cfg.train["epochs"]
board_path = cfg.meta["board_path"]
batch_size = cfg.train["batch_size"]
num_future_frame = cfg.model["future_frames"]
print_freq = cfg.train["print_frequency"]
img_size = cfg.model["input_size"]

file_identifier = f'{batch_size}_{epochs}_{img_size[0]}{img_size[1]}'


cfg.log_dict()
generator=nn.DataParallel(Generator(cfg.model,device))
discriminator=nn.DataParallel(Discriminator(cfg.model))
unet=unet_model()


# In[8]:


#CHANGE PATH 
FFP_DATASET_PATH = "hidden" ### CHANGE PATH TO THE HIDDEN DATASET
FFP_wt_path="frame_pred/model/model_4_50_80120.pt"
SM_wt_path="segmentation/segmentation_model.pth"


# In[9]:


FFP_state_dict = torch.load(FFP_wt_path,map_location="cpu")
SM_state_dict=torch.load(SM_wt_path, map_location="cpu")
gen_dic=FFP_state_dict["generator_state_dict"]
#discrim_dic=FFP_state_dict["discriminator_state_dict"]
unet_dic=SM_state_dict["model_state_dict"]


# In[10]:


generator.load_state_dict(gen_dic)
#discriminator.load_state_dict(discrim_dic)
unet.load_state_dict(unet_dic)
generator=generator.to(device)
#discriminator=discriminator.to(device)
unet=unet.to(device)


# In[11]:



class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get the list of folder names in the root directory
        self.folder_names = os.listdir(self.root_dir)
        self.folder_names.sort(key= lambda i: int(i.lstrip('video_')))
        
    def __len__(self):
        # Return the number of folders in the root directory
        return len(self.folder_names)
    
    def __getitem__(self, index):
        # Get the folder name corresponding to the given index
        folder_name = self.folder_names[index]
        print(folder_name)
        
        # Get the list of image filenames in the folder
        image_filenames = [i for i in os.listdir(os.path.join(self.root_dir, folder_name)) 
                           if i.endswith('.png')]
        #mask_filenames = [i for i in os.listdir(os.path.join(self.root_dir, folder_name)) 
        #                    if i.endswith('.npy')]
        image_filenames.sort(key= lambda i: int(i.lstrip('image_').rstrip('.png')))
        
        # Load the input images and target images into separate tensors
        input_images = []
        target_images = []
        for i, image_filename in enumerate(image_filenames):
            image_path = os.path.join(self.root_dir, folder_name, f"image_{i}.png")
            image = Image.open(image_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            input_images.append(image)
        #for i, mask_filename in enumerate(image_filenames):
            #mask_path = os.path.join(self.root_dir, folder_name, f"mask.npy")
            #masks = np.load(mask_path)
            #mask=masks[21]
        
        input_tensor = torch.stack(input_images)
        
        return input_tensor#, mask


# In[12]:


from torchvision import transforms
from torch.utils.data import DataLoader

t1 = transforms.Compose([
    transforms.Resize((80, 120)),
    transforms.ToTensor(),
])

t2=transforms.Compose([transforms.Resize((160,240)),transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),])
test_dataset = TestDataset(root_dir=f'{FFP_DATASET_PATH}', transform=t1)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# In[13]:


generator.eval()
#discriminator.eval()
unet.eval()


# In[23]:


answer_masks=[]
true_mask = []
i=0
with torch.no_grad():
    #for data , mask_r in test_dataloader:
    for data in test_dataloader:
        i+=1
        #if (i==5):
        #    break;
        data=data.to(device)
        pred_future_frames = generator(data, future=11)
        target_frames=pred_future_frames[:,21,:,:,:]
        target_frames=t2(target_frames)
        softmax = nn.Softmax(dim=1)
        predicted_mask = torch.argmax(softmax(unet(target_frames)),axis=1).squeeze(0)
        answer_masks.append(predicted_mask)
        #mask_r = mask_r.squeeze(0)
        #true_mask.append(mask_r)
        print(i, predicted_mask.shape)
        #print(i, mask_r.shape, predicted_mask.shape)
answer_masks=torch.stack(answer_masks,dim=0).to('cpu')
answer_masks=answer_masks.numpy()
np.save('to_Submit_answer_mask.npy',answer_masks)

#true_mask=torch.stack(true_mask,dim=0).to('cpu')
#true_mask=true_mask.numpy()
#np.save('true_mask.npy',true_mask)

print(answer_masks.shape)#, true_mask.shape)

#jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
#print(jaccard(torch.Tensor(true_mask), torch.Tensor(answer_masks)))
