import numpy as np
import torch
import torch.nn as nn
from args_fusion import args
import CBAM
import CA
import SA
from repvgg import RepVGGBlock,repvgg_model_convert
import utils


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = torch.nn.functional.leaky_relu(out, 0.01)
        return out

class ConvLayer_out(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer_out, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = torch.nn.functional.tanh(out)
        return out


# Shuffle network
class Repvgg_net(nn.Module):
    def __init__(self, input_nc=args.input_nc, output_nc=args.output_nc):
        super(Repvgg_net, self).__init__()
        nb_filter = [16, 32, 64, 32, 16]
        kernel_size = 3
        stride = 1

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool7 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

        #adjust H
        self.ir_encoder_H_conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.ir_encoder_H_conv2 = ConvLayer(nb_filter[0]*2, nb_filter[1], kernel_size, stride)
        self.vi_encoder_H_conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.vi_encoder_H_conv2 = ConvLayer(nb_filter[0]*2, nb_filter[1], kernel_size, stride)

        self.decoder_H_conv1 = ConvLayer(nb_filter[2]*2 + nb_filter[0]*2, nb_filter[1], kernel_size, stride)
        self.decoder_H_conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.decoder_H_conv3 = ConvLayer(nb_filter[1], input_nc, kernel_size, stride)

        self.ir_sa1 = SA.SABlock(kernel_size=3)
        self.ir_sa2 = SA.SABlock(kernel_size=7)
        self.vi_sa1 = SA.SABlock(kernel_size=3)
        self.vi_sa2 = SA.SABlock(kernel_size=7)

        self.sigmoid = nn.Sigmoid()

        self.MaxPool2d = nn.MaxPool2d(2)
        self.encoder_L_conv1_1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.encoder_L_conv1_2 = ConvLayer(nb_filter[0], nb_filter[0], kernel_size, stride)

        self.encoder_L_conv2_1 = ConvLayer(nb_filter[0], nb_filter[1], kernel_size, stride)
        self.encoder_L_conv2_2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)

        self.encoder_L_conv3_1 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.encoder_L_conv3_2 = ConvLayer(nb_filter[2], nb_filter[2], kernel_size, stride)

        self.encoder_L_conv4_1 = ConvLayer(nb_filter[2], nb_filter[2], kernel_size, stride)

        self.decoder_L_conv3_1 = ConvLayer(nb_filter[2]*2, nb_filter[2], kernel_size, stride)
        self.decoder_L_conv3_2 = ConvLayer(nb_filter[2], nb_filter[1], kernel_size, stride)

        self.decoder_L_conv2_1 = ConvLayer(nb_filter[1]*2, nb_filter[1], kernel_size, stride)
        self.decoder_L_conv2_2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.decoder_L_conv2_3 = ConvLayer(nb_filter[1], nb_filter[0], kernel_size, stride)

        self.decoder_L_conv1_1 = ConvLayer(nb_filter[0]*2, nb_filter[0], kernel_size, stride)
        self.decoder_L_conv1_2 = ConvLayer(nb_filter[0], nb_filter[0], kernel_size, stride)
        self.decoder_L_conv1_3 = ConvLayer(nb_filter[0], nb_filter[0], kernel_size, stride)

        self.decoder_up3 = nn.ConvTranspose2d(nb_filter[2], nb_filter[2], 2, stride=2)
        self.decoder_up2 = nn.ConvTranspose2d(nb_filter[1], nb_filter[1], 2, stride=2)
        self.decoder_up1 = nn.ConvTranspose2d(nb_filter[0], nb_filter[0], 2, stride=2)

        self.decoder_L_out1 = ConvLayer(nb_filter[0], nb_filter[0], kernel_size, stride)
        self.decoder_L_out2 = ConvLayer_out(nb_filter[0], output_nc, 1, stride)


        self.CA1 = CA.CA(nb_filter[0], reduction=4, kernel_size=kernel_size)
        self.CA2 = CA.CA(nb_filter[1], reduction=4, kernel_size=kernel_size)
        self.CA3 = CA.CA(nb_filter[2], reduction=4, kernel_size=kernel_size)
        self.CA4 = CA.CA(nb_filter[2], reduction=4, kernel_size=kernel_size)

    def encoder_H(self, ir_H, vi_H, ir, vi):
        ir_H_f1 = self.ir_encoder_H_conv1(ir_H)
        W_ir_H_f1 = self.ir_sa1(ir_H_f1)
        vi_H_f1 = self.vi_encoder_H_conv1(vi_H)
        W_vis_H_f1 = self.vi_sa1(vi_H_f1)

        ir_H_f2 = ir_H_f1 + ir_H_f1 * self.sigmoid(2*W_ir_H_f1 - W_vis_H_f1)
        vi_H_f2 = vi_H_f1 + vi_H_f1 * self.sigmoid(2*W_vis_H_f1 - W_ir_H_f1)

        ir_H_f2 = torch.cat((ir_H_f1, ir_H_f2), dim=1)
        vi_H_f2 = torch.cat((vi_H_f1, vi_H_f2), dim=1)

        ir_H_f2 = self.ir_encoder_H_conv2(ir_H_f2)
        W_ir_H_f2 = self.ir_sa2(ir_H_f2)
        vi_H_f2 = self.vi_encoder_H_conv2(vi_H_f2)
        W_vis_H_f2 = self.vi_sa2(vi_H_f2)

        ir_H_f3 = ir_H_f2+ ir_H_f2*self.sigmoid(2*W_ir_H_f2 - W_vis_H_f2)
        vi_H_f3 = vi_H_f2+ vi_H_f2*self.sigmoid(2*W_vis_H_f2 - W_ir_H_f2)

        ir_H_f3 = torch.cat((ir_H_f3, ir_H_f2,ir_H_f1), dim=1)
        vi_H_f3 = torch.cat((vi_H_f3, vi_H_f2,vi_H_f1), dim=1)

        f_H0 = torch.cat((ir_H_f3, vi_H_f3), dim=1)

        f_H1 = self.decoder_H_conv1(f_H0)
        f_H2 = self.decoder_H_conv2(f_H1)
        f_H3 = self.decoder_H_conv3(f_H2)

        return [f_H3]

    def encoder_L(self, ir_L, vi_L, ir_map, vi_map):
        ir_L_enf1 = self.encoder_L_conv1_1(ir_L)
        ir_L_enf1 = self.encoder_L_conv1_2(ir_L_enf1)

        ir_L_enf2 = self.MaxPool2d(ir_L_enf1)

        ir_L_enf2 = self.encoder_L_conv2_1(ir_L_enf2)
        ir_L_enf2 = self.encoder_L_conv2_2(ir_L_enf2)

        ir_L_enf3 = self.MaxPool2d(ir_L_enf2)

        ir_L_enf3 = self.encoder_L_conv3_1(ir_L_enf3)
        ir_L_enf3 = self.encoder_L_conv3_2(ir_L_enf3)

        ir_L_enf4 = self.MaxPool2d(ir_L_enf3)
        ir_L_enf4 = self.encoder_L_conv4_1(ir_L_enf4)

        #vis
        vi_L_enf1 = self.encoder_L_conv1_1(vi_L)
        vi_L_enf1 = self.encoder_L_conv1_2(vi_L_enf1)

        vi_L_enf2 = self.MaxPool2d(vi_L_enf1)

        vi_L_enf2 = self.encoder_L_conv2_1(vi_L_enf2)
        vi_L_enf2 = self.encoder_L_conv2_2(vi_L_enf2)

        vi_L_enf3 = self.MaxPool2d(vi_L_enf2)

        vi_L_enf3 = self.encoder_L_conv3_1(vi_L_enf3)
        vi_L_enf3 = self.encoder_L_conv3_2(vi_L_enf3)

        vi_L_enf4 = self.MaxPool2d(vi_L_enf3)
        vi_L_enf4 = self.encoder_L_conv4_1(vi_L_enf4)

        #map
        ir_map = torch.nn.functional.avg_pool2d(ir_map, kernel_size=19, stride=1, padding=9)
        vi_map = torch.nn.functional.avg_pool2d(vi_map, kernel_size=19, stride=1, padding=9)
        ir_map = torch.nn.functional.avg_pool2d(ir_map, kernel_size=19, stride=1, padding=9)
        vi_map = torch.nn.functional.avg_pool2d(vi_map, kernel_size=19, stride=1, padding=9)
        ir_map2 = self.MaxPool2d(ir_map)
        ir_map4 = self.MaxPool2d(ir_map2)
        ir_map8 = self.MaxPool2d(ir_map4)

        vi_map2 = self.MaxPool2d(vi_map)
        vi_map4 = self.MaxPool2d(vi_map2)
        vi_map8 = self.MaxPool2d(vi_map4)


        ##fusion
        W_vi_L_enf4 = self.CA4(vi_L_enf4)
        W_ir_L_enf4 = self.CA4(ir_L_enf4)
        vi_L_enf4 = vi_L_enf4 + vi_L_enf4 * self.sigmoid(2 * W_vi_L_enf4 - W_ir_L_enf4)
        ir_L_enf4 = ir_L_enf4 + ir_L_enf4 * self.sigmoid(2 * ir_L_enf4 - vi_L_enf4)
        f_enf4 = (vi_L_enf4 * (vi_map8) + ir_L_enf4 * (ir_map8))

        W_vi_L_enf3 = self.CA3(vi_L_enf3)
        W_ir_L_enf3 = self.CA3(ir_L_enf3)
        vi_L_enf3 = vi_L_enf3 + vi_L_enf3 * self.sigmoid(2 * W_vi_L_enf3 - W_ir_L_enf3)
        ir_L_enf3 = ir_L_enf3 + ir_L_enf3 * self.sigmoid(2 * ir_L_enf3 - vi_L_enf3)
        f_enf3 = (vi_L_enf3 * (vi_map4) + ir_L_enf3 * (ir_map4))

        W_vi_L_enf2 = self.CA2(vi_L_enf2)
        W_ir_L_enf2 = self.CA2(ir_L_enf2)
        vi_L_enf2 = vi_L_enf2 + vi_L_enf2 * self.sigmoid(2 * W_vi_L_enf2 - W_ir_L_enf2)
        ir_L_enf2 = ir_L_enf2 + ir_L_enf2 * self.sigmoid(2 * ir_L_enf2 - vi_L_enf2)
        f_enf2 = (vi_L_enf2 * (vi_map2) + ir_L_enf2 * (ir_map2))

        W_vi_L_enf1 = self.CA1(vi_L_enf1)
        W_ir_L_enf1 = self.CA1(ir_L_enf1)
        vi_L_enf1 = vi_L_enf1 + vi_L_enf1 * self.sigmoid(2 * W_vi_L_enf1 - W_ir_L_enf1)
        ir_L_enf1 = ir_L_enf1 + ir_L_enf1 * self.sigmoid(2 * ir_L_enf1 - W_vi_L_enf1)
        f_enf1 = (vi_L_enf1 * (vi_map) + ir_L_enf1 * (ir_map))

        #up
        f_up3 = self.decoder_up3(f_enf4)
        f_up3 = torch.nn.functional.leaky_relu(f_up3, 0.01)
        f_L_def3 = self.decoder_L_conv3_1(torch.cat((f_up3, f_enf3), dim=1))
        f_L_def3 = self.decoder_L_conv3_2(f_L_def3)

        # up
        f_up2 = self.decoder_up2(f_L_def3)
        f_up2 = torch.nn.functional.leaky_relu(f_up2, 0.01)
        f_L_def2 = self.decoder_L_conv2_1(torch.cat((f_up2, f_enf2), dim=1))
        f_L_def2 = self.decoder_L_conv2_2(f_L_def2)
        f_L_def2 = self.decoder_L_conv2_3(f_L_def2)

        # up
        f_up1 = self.decoder_up1(f_L_def2)
        f_up1 = torch.nn.functional.leaky_relu(f_up1, 0.01)
        f_L_def1 = self.decoder_L_conv1_1(torch.cat((f_up1, f_enf1), dim=1))
        f_L_def1 = self.decoder_L_conv1_2(f_L_def1)
        f_L_def1 = self.decoder_L_conv1_3(f_L_def1)

        #out
        out = self.decoder_L_out1(f_L_def1)
        out = self.decoder_L_out2(out)

        return [out]
