import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from copy import deepcopy

import torchvision.models as models
from change_utils import weights_init, conv2DBatchNormRelu, conv2DRelu, deconv2DBatchNormRelu, deconv2DRelu
from utils.softargmax import SoftArgmax2D, create_meshgrid
from utils.dataset import augment_data, create_images_dict
from utils.image_utils import create_gaussian_heatmap_template, create_dist_mat, \
	preprocess_image_for_segmentation, pad, resize
from utils.dataloader import SceneDataset, scene_collate
from test import evaluate
from train import train


class StyleModulator(nn.Module):
	def __init__(self, sizes):
		"""
		Additional style modulator for efficient fine-tuning
		"""		
		from ddf import DDFPack
		super(StyleModulator, self).__init__()
		tau = 0.5
		self.modulators = nn.ModuleList(
			[DDFPack(s) for s in sizes + [sizes[-1]]]
		)

	def forward(self, x):
		stylized = []
		for xi, layer in zip(x, self.modulators):
			stylized.append(layer(xi))
		return stylized

class ResNetEncoder(nn.Module):
    def __init__(self, num_layers=18):
        super(ResNetEncoder, self).__init__()
        if num_layers == 18:
            self.resnet = models.resnet18(pretrained=True)
            self.num_channels = [64, 128, 256, 512]
        elif num_layers == 34:
            self.resnet = models.resnet34(pretrained=True)
            self.num_channels = [64, 128, 256, 512]
        elif num_layers == 50:
            self.resnet = models.resnet50(pretrained=True)
            self.num_channels = [256, 512, 1024, 2048]
        elif num_layers == 101:
            self.resnet = models.resnet101(pretrained=True)
            self.num_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError("Invalid number of layers for ResNet")

    def forward(self, x):
        features = []
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        features.append(x)

        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        features.append(x)

        x = self.resnet.layer2(x)
        features.append(x)

        x = self.resnet.layer3(x)
        features.append(x)

        x = self.resnet.layer4(x)
        features.append(x)

        return features



class ResNetDecoder(nn.Module):
    def __init__(self, num_channels=[512, 256, 128, 64], output_len=30):
        super(ResNetDecoder, self).__init__()
        self.num_channels = num_channels
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(out_channels_),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(out_channels_),
                nn.ReLU(inplace=True)
            )
            for in_channels_, out_channels_ in zip(num_channels[:-1], num_channels[1:])
        ])
        self.predictor = nn.Conv2d(num_channels[-1], output_len * 2, kernel_size=1)

    def forward(self, features):
        x = features[-1]
        for i in range(len(self.decoder)):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            skip_connection = features[-i-2]
            x = torch.cat([x, skip_connection], dim=1)
            x = self.decoder[i](x)
        x = self.predictor(x)
        return x
 

class YNetTorch(nn.Module):
    def __init__(self, num_channels=[64, 128, 256, 512], output_len=30):
        super(YNetTorch, self).__init__()
        self.encoder = ResNetEncoder()
        self.decoder = ResNetDecoder(num_channels=num_channels, output_len=output_len)

    def forward(self, x):
        features = self.encoder(x)
        prediction = self.decoder(features)
        return prediction


class YNet:
	def __init__(self, obs_len, pred_len, params):
		"""
		Ynet class, following a sklearn similar class structure
		:param obs_len: observed timesteps
		:param pred_len: predicted timesteps
		:param params: dictionary with hyperparameters
		"""
		self.obs_len = obs_len
		self.pred_len = pred_len
		self.division_factor = 2 ** len(params['encoder_channels'])

		model = YNetTorch(num_channels=[64, 128, 256, 512], output_len=30).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
		criterion = nn.MSELoss()

		for epoch in range(num_epochs):
			train_loss = train(model, train_loader, optimizer, criterion, device)
			test_loss = evaluate(model,test_loader,criterion,cuda)
			print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(epoch+1, train_loss, test_loss))

	def train(model, train_loader, optimizer, criterion, device):
		model.train()
		train_loss = 0
		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = data.to(device), target.to(device)
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()

		return train_loss / len(train_loader)


	def evaluate(model, test_loader, criterion, device):
		model.eval()
		test_loss = 0
		with torch.no_grad():
			for data, target in test_loader:
				data, target = data.to(device), target.to(device)
				output = model(data)
				test_loss += criterion(output, target).item()
		return test_loss / len(test_loader)


def load(self, path):
	print(self.model.load_state_dict(torch.load(path), strict=False))

def save(self, path):
		torch.save(self.model.state_dict(), path)
