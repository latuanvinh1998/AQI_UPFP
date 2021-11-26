from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid
from torch.nn import Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
import math

class Neural_Net(Module):
	def __init__(self, in_features_img, num_cls):
		super(Neural_Net, self).__init__()
		self.nn_1 = Linear(in_features_img, 1024)
		self.nn_2 = Linear(1024, 512)
		self.prelu_1 = PReLU(1024)
		self.prelu_2 = PReLU(512)
		self.nn_last = Linear(512, num_cls)

	def forward(self, img):
		emb_1 = self.nn_1(img)
		emb_1 = self.prelu_1(emb_1)
		emb_2 = self.nn_2(emb_1)
		emb = self.prelu_2(emb_2)

		out = self.nn_last(emb)
		return out
