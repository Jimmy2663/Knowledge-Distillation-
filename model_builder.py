"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn 
import torch.nn.functional as F

class VGG16(nn.Module):
    """Creates the VGG16 architecture.

    Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
    """
    def __init__(self, input_shape: int,  output_shape: int, dropout: float) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=64, 
                    kernel_size=3, 
                    stride=1, 
                    padding="same" ),  
          nn.ReLU(),
          nn.Conv2d(in_channels=64, 
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding="same"),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
        )
        self.conv_block_2 = nn.Sequential(
          nn.Conv2d(64, 128, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.Conv2d(128,128, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.MaxPool2d(2)
        )
        self.conv_block_3 = nn.Sequential(
          nn.Conv2d(128, 256, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.Conv2d(256, 256, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.Conv2d(256, 256, kernel_size=3, padding="same"),
          nn.ReLU(),          
          nn.MaxPool2d(2)
        )
        self.conv_block_4 = nn.Sequential(
          nn.Conv2d(256, 512, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.Conv2d(512, 512, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.Conv2d(512, 512, kernel_size=3, padding="same"),
          nn.ReLU(),          
          nn.MaxPool2d(2)
        )
        self.conv_block_5 = nn.Sequential(
          nn.Conv2d(512, 512, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.Conv2d(512, 512, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.Conv2d(512, 512, kernel_size=3, padding="same"),
          nn.ReLU(),          
          nn.MaxPool2d(2)
        )                         
        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=512*7*7,
                    out_features=4096
                    ),
          nn.ReLU(),
          nn.Dropout(p=dropout),
          nn.Linear(in_features=4096,
                    out_features=4096
                    ),
          nn.ReLU(),
          nn.Dropout(p=dropout),
          nn.Linear(in_features=4096,
                    out_features=output_shape)           
        )
    
    def forward(self, x: torch.Tensor):
        #print(x.shape)
        x = self.conv_block_1(x)
        #print(x.shape)            
        x = self.conv_block_2(x)
        #print(x.shape)
        x = self.conv_block_3(x)
        #print(x.shape)
        x = self.conv_block_4(x)
        #print(x.shape)
        x = self.conv_block_5(x)
        #print(x.shape)                        
        x = self.classifier(x)
        return x
        # return self.classifier(self.block_2(self.block_1(x))) # <- leverage the benefits of operator fusion



class BasicBlock(nn.Module):
    """
    Implements a basic residual block for ResNet18.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride for the first convolutional layer. Default: 1.
        downsample (nn.Module, optional): Downsampling layer to match dimensions. Default: None.

    Forward Input:
        x (Tensor): Input tensor of shape (batch_size, in_channels, H, W).

    Forward Output:
        out (Tensor): Output tensor after applying the residual block.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # First convolutional layer with batch normalization and ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Second convolutional layer with batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Downsampling layer for shortcut connection, if needed
        self.downsample = downsample

    def forward(self, x):
        """
        Forward pass for the basic residual block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after residual addition and activation.
        """
        identity = x  # Save input for the shortcut path

        # Main path: conv -> BN -> ReLU -> conv -> BN
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # If dimensions mismatch, apply downsampling to the shortcut
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add shortcut to the main path
        out += identity
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    """
    Implements the ResNet18 architecture.

    Args:
        num_classes (int): Number of output classes for classification.

    Forward Input:
        x (Tensor): Input tensor of shape (batch_size, 3, 224, 224).

    Forward Output:
        x (Tensor): Output logits of shape (batch_size, num_classes).
    """
    def __init__(self, num_classes=int):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # Initial convolutional layer: 7x7 kernel, stride 2, followed by batch norm and ReLU
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # Max pooling layer to reduce spatial dimensions
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four residual layers, each with two basic blocks
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # Adaptive average pooling to produce 1x1 spatial output
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer for classification
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Creates a sequential layer of residual blocks.

        Args:
            block (nn.Module): Block type to use (e.g., BasicBlock).
            out_channels (int): Number of output channels for the layer.
            blocks (int): Number of blocks to stack.
            stride (int, optional): Stride for the first block. Default: 1.

        Returns:
            nn.Sequential: Stacked residual blocks as a sequential module.
        """
        downsample = None
        # If input and output dimensions differ, add a downsampling layer
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block may downsample
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the ResNet18 model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output logits for each class.
        """
        # Initial convolution, batch norm, ReLU, and max pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Pass through four residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling and flatten
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # Fully connected layer for classification
        x = self.fc(x)

        return x


