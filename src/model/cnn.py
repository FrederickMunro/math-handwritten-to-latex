import torch
from torch.nn import Module, Conv2d, BatchNorm2d, Sequential, MaxPool2d
import torch.nn.functional as F


class ConvNormRelu(Module):
    """
    A stack of a convolutional layer, batch normalization, and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (tuple or int): Size of the convolutional kernel.
        dropout (bool, optional): Whether to apply dropout or not. Default is False.
        p (float, optional): Dropout probability. Default is 0.2.

    Attributes:
        conv (torch.nn.Conv2d): Convolutional layer.
        batchNorm (torch.nn.BatchNorm2d): Batch normalization layer.
        dropout (bool): Whether to apply dropout or not.
        p (float): Dropout probability.

    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout=False, p=0.2):
        """
        Initialize the ConvNormRelu module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple or int): Size of the convolutional kernel.
            dropout (bool): Whether to apply dropout or not.
            p (float): Dropout probability.
        """
        super(ConvNormRelu, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size) # Convolutional layer
        self.batchNorm = BatchNorm2d(out_channels) # Batch normalization
        self.dropout = dropout
        self.p = p

    def forward(self, batch):
        """
        Forward pass of the ConvNormRelu module.

        Args:
            batch (torch.Tensor): Input batch.

        Returns:
            torch.Tensor: Output of the module after applying convolution, batch normalization,
                          and ReLU activation.
        """
        if self.dropout:
            # Dropout with specified probability
            batch = F.dropout(batch, p=self.p)
        # ReLU activation after batch normalization and convolution
        return F.relu(self.batchNorm(self.conv(batch)))


class CNN(Module):
    """
    CNN for feature extraction.

    Args:
        device (torch.device): Device on which the model is run.

    Attributes:
        layer1 (torch.nn.Sequential): First layer of the CNN model.
        layer2 (torch.nn.Sequential): Second layer of the CNN model.
        layer3 (torch.nn.Sequential): Third layer of the CNN model.
        layer4 (ConvNormRelu): Fourth layer of the CNN model.

    Examples:
        # Create a CNN instance
        >>> model = CNN(device)

        # Forward pass
        >>> output = model(batch)
    """
    def __init__(self, device):
        """
        Initialize the CNN module.

        Args:
            device (torch.device): Device on which the model is run.

        """
        super(CNN, self).__init__()

        # Layer 1: Convolutional, BatchNorm, ReLU, MaxPool
        self.layer1 = Sequential(
            ConvNormRelu(1, 64, (3, 3)),
            MaxPool2d((2, 2))
        )

        # Layer 2: Convolutional, BatchNorm, ReLU, MaxPool
        self.layer2 = Sequential(
            ConvNormRelu(64, 128, (3, 3)),
            MaxPool2d((2, 2))
        )

        # Layer 3: Multiple Convolutional, BatchNorm, ReLU, MaxPool
        self.layer3 = Sequential(
            ConvNormRelu(128, 256, (3, 3)),
            ConvNormRelu(256, 256, (3, 3)),
            MaxPool2d((1, 2)),
            ConvNormRelu(256, 512, (3, 3)),
            MaxPool2d((2, 1))
        )

        # Layer 4: Convolutional, BatchNorm, ReLU
        self.layer4 = ConvNormRelu(512, 512, (3, 3)).to(device)

    def forward(self, batch):
        """
        Forward pass through the CNN model.

        Args:
            batch (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Output tensor after passing through the CNN model.
        """
        # Forward pass through each layer
        out = self.layer1(batch)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
