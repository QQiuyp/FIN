import numpy as np
import torch.nn as nn

class ConvRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1, init_zero=False):
        super(ConvRelu, self).__init__()

        self.init_zero = init_zero
        if self.init_zero:
            self.layers = nn.Conv2d(channels_in, channels_out, 3, stride, padding=1)

        else:
            self.layers = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        return self.layers(x)


class ConvTRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=2):
        super(ConvTRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=stride, padding=0),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class ExpandNet(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(ExpandNet, self).__init__()

        layers = [ConvTRelu(in_channels, out_channels)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = ConvTRelu(out_channels, out_channels)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, H=128, message_length=64, channels=32):
        super(Encoder, self).__init__()

        stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))

        self.message_pre_layer = nn.Sequential(
            ConvRelu(1, channels),
            ExpandNet(channels, channels, blocks=stride_blocks),
            ConvRelu(channels, 1, init_zero=True),
        )

    def forward(self, message):
        size = int(np.sqrt(message.shape[1]))
        message_image = message.view(-1, 1, size, size)
        message = self.message_pre_layer(message_image)
        return message






