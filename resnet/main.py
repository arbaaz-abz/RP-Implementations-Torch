import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
	def __init__(self, in_channels, out_channels, identity_downsample, stride=1):
		super(Block, self).__init__()
		self.expansion = 4
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
		self.bn1 = nn.BatchNorm2d(out_channels)

		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
		self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

		self.relu = nn.ReLU()

		self.identity_downsample = identity_downsample


	def forward(self, x):
		identity = x

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)

		x = self.conv3(x)
		x = self.bn3(x)

		if self.identity_downsample is not None:
			identity = self.identity_downsample(identity)

		x += identity
		x = self.relu(x)

		return x

class ResNet(nn.Module):
	# Input image size - 224 x 224 x image_channels
	def __init__(self, layers, image_channels, num_classes):
		super(ResNet, self).__init__()

		self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)  # Ensures an output size of 112 x 112
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.prev_Lout_channels = 64

		self.layer1 = self._make_layer_(num_blocks = layers[0], init_channels = 64, stride = 1)
		self.layer2 = self._make_layer_(num_blocks = layers[1], init_channels = 128, stride = 2)
		self.layer3 = self._make_layer_(num_blocks = layers[2], init_channels = 256, stride = 2)
		self.layer4 = self._make_layer_(num_blocks = layers[3], init_channels = 512, stride = 2)

		self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(self.prev_Lout_channels, num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		print(x.shape)
		x = self.relu(x)
		x = self.maxpool(x)
		print(x.shape)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		print(x.shape)
		x = self.avgPool(x)
		print(x.shape)
		x = self.fc(x.squeeze())
		print(x.shape)
		return x


	def _make_layer_(self, num_blocks, init_channels, stride):
		identity_downsample = None
		layers = []
		
		# Apply residual connection for the first block in a layer 
		if stride != 1 or self.prev_Lout_channels != init_channels * 4:
			identity_downsample = nn.Sequential(nn.Conv2d(self.prev_Lout_channels, init_channels * 4, kernel_size = 1, stride=stride),
												nn.BatchNorm2d(init_channels * 4))

		layers.append(Block(self.prev_Lout_channels, init_channels, identity_downsample, stride))
		self.prev_Lout_channels = init_channels * 4

		for i in range(num_blocks - 1):
			layers.append(Block(self.prev_Lout_channels, init_channels, None, 1))
		
		return nn.Sequential(*layers)


if __name__ == "__main__":
	resnet = ResNet([3, 4, 6, 3], 3, 1000)
	random_image = torch.randn([10, 3, 224, 224])
	result = resnet.forward(random_image)
	probabilities = F.softmax(result, dim=1)
	print(torch.argmax(probabilities, dim=1))

	resnet = ResNet([3, 4, 23, 3], 3, 1000)
	random_image = torch.randn([10, 3, 224, 224])
	result = resnet.forward(random_image)
	probabilities = F.softmax(result, dim=1)
	print(torch.argmax(probabilities, dim=1))

	resnet = ResNet([3, 8, 36, 3], 3, 1000)
	random_image = torch.randn([10, 3, 224, 224])
	result = resnet.forward(random_image)
	probabilities = F.softmax(result, dim=1)
	print(torch.argmax(probabilities, dim=1))





