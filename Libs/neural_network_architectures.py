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
			nn.Conv2d(1, 64, 3, 2, 1),  # in_channels, out_channels, kernel_size, stride, padding
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

