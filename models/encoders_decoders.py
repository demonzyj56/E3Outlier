from keras.models import Model
from keras.layers import Input, Conv2D, Activation, BatchNormalization, Flatten, Dense, Conv2DTranspose, Reshape
from utils import get_channels_axis
import torch
import torch.nn as nn


def conv_encoder(input_side=32, n_channels=3, representation_dim=256, representation_activation='tanh',
                 intermediate_activation='relu'):
    nf = 64
    input_shape = (n_channels, input_side, input_side) if get_channels_axis() == 1 else (input_side, input_side,
                                                                                         n_channels)

    x_in = Input(shape=input_shape)
    enc = x_in

    # downsample x0.5
    enc = Conv2D(nf, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc)
    enc = BatchNormalization(axis=get_channels_axis())(enc)
    enc = Activation(intermediate_activation)(enc)

    # downsample x0.5
    enc = Conv2D(nf * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc)
    enc = BatchNormalization(axis=get_channels_axis())(enc)
    enc = Activation(intermediate_activation)(enc)

    # downsample x0.5
    enc = Conv2D(nf * 4, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc)
    enc = BatchNormalization(axis=get_channels_axis())(enc)
    enc = Activation(intermediate_activation)(enc)

    if input_side == 64:
        # downsample x0.5
        enc = Conv2D(nf * 8, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc)
        enc = BatchNormalization(axis=get_channels_axis())(enc)
        enc = Activation(intermediate_activation)(enc)

    enc = Flatten()(enc)
    rep = Dense(representation_dim, activation=representation_activation)(enc)

    return Model(x_in, rep)


def conv_decoder(output_side=32, n_channels=3, representation_dim=256, activation='relu'):
    nf = 64

    rep_in = Input(shape=(representation_dim,))

    g = Dense(nf * 4 * 4 * 4)(rep_in)
    g = BatchNormalization(axis=-1)(g)
    g = Activation(activation)(g)

    conv_shape = (nf * 4, 4, 4) if get_channels_axis() == 1 else (4, 4, nf * 4)
    g = Reshape(conv_shape)(g)

    # upsample x2
    g = Conv2DTranspose(nf * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(g)
    g = BatchNormalization(axis=get_channels_axis())(g)
    g = Activation(activation)(g)

    # upsample x2
    g = Conv2DTranspose(nf, kernel_size=(3, 3), strides=(2, 2), padding='same')(g)
    g = BatchNormalization(axis=get_channels_axis())(g)
    g = Activation(activation)(g)

    if output_side == 64:
        # upsample x2
        g = Conv2DTranspose(nf, kernel_size=(3, 3), strides=(2, 2), padding='same')(g)
        g = BatchNormalization(axis=get_channels_axis())(g)
        g = Activation(activation)(g)

    # upsample x2
    g = Conv2DTranspose(n_channels, kernel_size=(3, 3), strides=(2, 2), padding='same')(g)
    g = Activation('tanh')(g)

    return Model(rep_in, g)

class CAE_pytorch(nn.Module):
    def __init__(self, in_channels = 3, rep_dim = 256):
        super(CAE_pytorch, self).__init__()
        nf = 64
        self.nf = nf

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(num_features=nf)
        self.enc_act1 = nn.ReLU(inplace=True)

        self.enc_conv2 = nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(num_features=nf * 2)
        self.enc_act2 = nn.ReLU(inplace=True)

        self.enc_conv3 = nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(num_features=nf * 4)
        self.enc_act3 = nn.ReLU(inplace=True)

        self.enc_fc = nn.Linear(nf * 4 * 4 * 4, rep_dim)
        self.rep_act = nn.Tanh()

        # Decoder
        self.dec_fc = nn.Linear(rep_dim, nf * 4 * 4 * 4)
        self.dec_bn0 = nn.BatchNorm1d(num_features=nf * 4 * 4 *4)
        self.dec_act0 = nn.ReLU(inplace=True)

        self.dec_conv1 = nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(num_features=nf * 2)
        self.dec_act1 = nn.ReLU(inplace=True)

        self.dec_conv2 = nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(num_features=nf)
        self.dec_act2 = nn.ReLU(inplace=True)

        self.dec_conv3 = nn.ConvTranspose2d(in_channels=nf, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.output_act = nn.Tanh()

    def encode(self, x):
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
        return rep

    def decode(self, rep):
        x = self.dec_act0(self.dec_bn0(self.dec_fc(rep)))
        x = x.view(-1, self.nf * 4, 4, 4)
        x = self.dec_act1(self.dec_bn1(self.dec_conv1(x)))
        x = self.dec_act2(self.dec_bn2(self.dec_conv2(x)))
        x = self.output_act(self.dec_conv3(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))





