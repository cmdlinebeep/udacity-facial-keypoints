## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
#         # Inputs are 1-channel (grayscale), and 224 x 224 square  
#         # Filter equation:  (W - F)/S + 1
        
#         self.conv1 = nn.Conv2d(1, 32, 5)    # (224 - 5)/1 + 1 = 220.  Output size: (32, 220, 220)  (filter outputs, height, width)
#         self.conv2 = nn.Conv2d(32, 16, 5)   # (220 - 5)/1 + 1 = 216   (16, 216, 216)
#         self.pool1 = nn.MaxPool2d(2, 2)     # (16, 108, 108)  # divide height and width by 2 each since stride of 2
                
# #         self.batchn1 = nn.BatchNorm1d(
#         self.drop1 = nn.Dropout(p=0.4)
        
# #         self.fc1 = nn.Linear(512, 256)  # No!  Dimensional mismatch.  Input units has to match volumetric cube above
#         self.fc1 = nn.Linear(16*108*108, 256)
#         self.drop2 = nn.Dropout(p=0.4)
#         self.fc2 = nn.Linear(256, 136)           

        # Running out of memory or taking too long, try smaller input images
    
        # Inputs are 1-channel (grayscale), and 128 x 128 square  
        # Filter equation:  (W - F)/S + 1

        # Now that running locally on my GPU, moved back to 224 inputs instead of 128!
        
        self.conv1 = nn.Conv2d(1, 32, 5)    # (224 - 5)/1 + 1 = 220.  Output size: (32, 220, 220)  (filter outputs, height, width)
        self.pool1 = nn.MaxPool2d(2, 2)     # (32, 110, 110)  # divide height and width by 2 each since stride of 2
        self.bn1 = nn.BatchNorm2d(32)       # Needs to match the number of outputs
        self.conv2 = nn.Conv2d(32, 64, 5)   # (110 - 5)/1 + 1 = 106   (64, 106, 106)
        self.pool2 = nn.MaxPool2d(2, 2)     # (64, 53, 53)  # divide height and width by 2 each since stride of 2
        self.bn2 = nn.BatchNorm2d(64)       # Needs to match number of outputs
        self.conv3 = nn.Conv2d(64, 128, 7)  # (53-7)/1 + 1 = 47.  (128, 47, 47)
        self.pool3 = nn.MaxPool2d(2, 2)     # (128, 23, 23) rounds down
        self.bn3 = nn.BatchNorm2d(128)
                
#         self.batchn1 = nn.BatchNorm1d(
               
        self.fc1 = nn.Linear(128*23*23, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 136)
        # self.fc2_bn = nn.BatchNorm1d(136)
        # self.fc2_drop = nn.Dropout(p=0.5)  # If Dropout and BN are both used, Dropout must be after all BN layers!
        # self.drop2 = nn.Dropout(p=0.4)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting 
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.bn3(x)
        
        # Need to flatten
        x = x.view(x.size(0), -1)
#         x = torch.flatten(x, 1)  # easier way.  flatten all dimensions except batch  ==> Didn't work!
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc1_bn(x)

        x = self.fc2(x)    # No ReLU after this one because regression (probably still could, since using relu not sigmoid)
        # x = self.fc2_bn(x)
        # x = self.fc2_drop(x)
        # x = self.drop2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
