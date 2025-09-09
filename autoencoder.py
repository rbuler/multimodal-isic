import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, base_filters=32, z_dim=128, dropout_p=0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, base_filters),
            nn.Dropout2d(dropout_p)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, base_filters*2),
            nn.Dropout2d(dropout_p)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, base_filters*4),
            nn.Dropout2d(dropout_p)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*8, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, base_filters*8),
            nn.Dropout2d(dropout_p)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(base_filters*8, base_filters*16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, base_filters*16),
            nn.Dropout2d(dropout_p)
        )

        self._out_h = base_filters*16*11*11
        self.fc_mu = nn.Linear(self._out_h, z_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        B, C, H, W = x.size()
        self._bottleneck_hw = (H, W)
        x = x.view(B, -1)
        z = self.fc_mu(x)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, out_channels=3, base_filters=32, z_dim=128):
        super().__init__()
        self.base_filters = base_filters
        self.fc = None  

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_filters*16, base_filters*8, 4, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, base_filters*8),

            nn.ConvTranspose2d(base_filters*8, base_filters*4, 4, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, base_filters*4),

            nn.ConvTranspose2d(base_filters*4, base_filters*2, 4, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, base_filters*2),

            nn.ConvTranspose2d(base_filters*2, base_filters, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, base_filters),

            nn.ConvTranspose2d(base_filters, out_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z, bottleneck_hw):
        B = z.size(0)
        H, W = bottleneck_hw
        if self.fc is None:
            self._init_size = self.deconv[0].in_channels * H * W
            self.fc = nn.Linear(z.size(1), self._init_size).to(z.device)
        x = self.fc(z)
        x = x.view(B, self.deconv[0].in_channels, H, W)
        x = self.deconv(x)
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=3, base_filters=32, z_dim=128, dropout_p=0.2):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, base_filters, z_dim, dropout_p)
        self.decoder = ConvDecoder(in_channels, base_filters, z_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z, self.encoder._bottleneck_hw)
        return recon, z
