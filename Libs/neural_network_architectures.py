import torch
import torch.nn as nn
import torch.nn.functional as TF
#-----------------------------------------------------------------------------------------#

# NOTE NeuralNetWithDropout (1D)
class NNWD(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(NNWD, self).__init__()
		self.fc1 = nn.Linear(input_dim, 305)
		self.fc2 = nn.Linear(305, 64)
		self.dropout = nn.Dropout(p=0.1)
		self.fc3 = nn.Linear(64, 69)
		self.fc4 = nn.Linear(69, output_dim)

	def forward(self, x):
		x = TF.relu(self.fc1(x))
		x = TF.relu(self.fc2(x))
		x = self.dropout(x)
		x = TF.relu(self.fc3(x))
		x = self.fc4(x)
		return x

class AlexNet(nn.Module):
	def __init__(self, output_dim):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, 3, 2, 1),  # in_channels, out_channels, kernel_size, stride, padding
			nn.MaxPool2d(2),  # kernel_size
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 192, 3, padding=1),
			nn.MaxPool2d(2),
			nn.ReLU(inplace=True),
			nn.Conv2d(192, 384, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.MaxPool2d(2),
			nn.ReLU(inplace=True)
		)
		self.classifier = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(256 * 2 * 2, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, output_dim),
		)

	def forward(self, x):
		x = self.features(x)
		h = x.view(x.shape[0], -1)
		x = self.classifier(h)
		return x, h

class ResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        block, layers, channels = config
        self.in_channels = channels[0]
        assert len(layers) == len(channels) == 3
        assert all([i == j*2 for i, j in zip(channels[1:], channels[:-1])])
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.layer1 = self.get_resnet_layer(block, layers[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, layers[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, layers[2], channels[2], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)
        
    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):
        layers = []
        if self.in_channels != channels:
            downsample = True
        else:
            downsample = False
        layers.append(block(self.in_channels, channels, stride, downsample))
        for i in range(1, n_blocks):
            layers.append(block(channels, channels))
        self.in_channels = channels
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        return x, h

class Identity(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        
    def forward(self, x):
        return self.f(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        if downsample:
            identity_fn = lambda x : TF.pad(x[:, :, ::2, ::2], [0, 0, 0, 0, in_channels // 2, in_channels // 2])
            downsample = Identity(identity_fn)
        else:
            downsample = None
        self.downsample = downsample
        
    def forward(self, x):
        i = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            i = self.downsample(i)
        x += i
        x = self.relu(x)
        return x

