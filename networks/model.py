import torch.nn as nn
import networks.pnasnet
import torch.nn.functional as F

def create_model(model_name, decoder_scale=1024):
    # Create model
    if model_name == 'PNASNet5Large':
        model = Model_PNASNet5Large(decoder_scale).cuda()
    return model

class Up(nn.Sequential):
    def __init__(self, num_input_channels, num_output_channels):
        super(Up, self).__init__()
        self.convA = nn.Conv2d(num_input_channels, num_output_channels, kernel_size=3, stride=1, padding=1)
        self.convB = nn.Conv2d(num_output_channels, num_output_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_up = F.interpolate(x, size=[x.size(2)*2, x.size(3)*2], mode='bilinear', align_corners=True)
        x_convA = self.relu(self.convA(x_up))
        x_convB = self.relu(self.convB(x_convA))

        return x_convB

class Model_PNASNet5Large(nn.Module):
    def __init__(self, decoder_scale = 1024):
        super(Model_PNASNet5Large, self).__init__()
        self.encoder = networks.pnasnet.pnasnet5large(num_classes=1000, pretrained='imagenet')

        num_channels_d32_in = 4320
        num_channels_d32_out = decoder_scale

        self.conv_d32 = nn.Conv2d(num_channels_d32_in, num_channels_d32_out, kernel_size=1, stride=1)

        self.up1 = Up(num_input_channels=num_channels_d32_out // 1, num_output_channels=num_channels_d32_out // 2)
        self.up2 = Up(num_input_channels=num_channels_d32_out // 2, num_output_channels=num_channels_d32_out // 4)
        self.up3 = Up(num_input_channels=num_channels_d32_out // 4, num_output_channels=num_channels_d32_out // 8)
        self.up4 = Up(num_input_channels=num_channels_d32_out // 8, num_output_channels=num_channels_d32_out // 16)
        self.up5 = Up(num_input_channels=num_channels_d32_out // 16, num_output_channels=num_channels_d32_out // 32)
        self.conv3 = nn.Conv2d(num_channels_d32_out // 32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        encoder_out = self.encoder.get_features_min(x)
        decoder_in = self.conv_d32(encoder_out)

        decoder_x2 = self.up1(decoder_in)
        decoder_x4 = self.up2(decoder_x2)
        decoder_x8 = self.up3(decoder_x4)
        decoder_x16 = self.up4(decoder_x8)
        decoder_x32 = self.up5(decoder_x16)
        output = self.conv3(decoder_x32)
        return output