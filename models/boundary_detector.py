import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y


def get_laplace(in_chan, out_chan):
    # conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    # nn.init.constant_(conv.weight, 1)
    # nn.init.constant_(conv.weight[0, 0, 1, 1], -8)
    # nn.init.constant_(conv.weight[0, 1, 1, 1], -8)
    # nn.init.constant_(conv.weight[0, 2, 1, 1], -8)
    filter = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ]).astype(np.float32)
    filter = filter.reshape((1, 1, 3, 3))
    filter = np.repeat(filter, in_chan, axis=1)
    filter = np.repeat(filter, out_chan, axis=0)

    filter = torch.from_numpy(filter)
    filter = nn.Parameter(filter, requires_grad=False)
    conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv.weight = filter
    laplacian = nn.Sequential(conv, nn.BatchNorm2d(out_chan))

    return laplacian


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input


def run_laplace(operator, x):
    out = operator(x)
    return torch.sigmoid(out) * x


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.block = nn.Sequential(
            ConvBNR(512 + 256, 512, 3),
            ConvBNR(512, 256, 3),
            ConvBNR(256, 128, 3),
            nn.Conv2d(128, 1, 1))

    def forward(self, x_fuse, x_depth):
        out = torch.cat((x_fuse, x_depth))
        out = self.block(out)

        return out


def gabor_gen(sigma, theta, Lambda, gamma, ksize, cos_or_sin):
    """cos_or_sin=1 return cos square_wave."""
    """cos_or_sin=0 return sin square_wave."""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    xmax = int(ksize[0] / 2)
    ymax = int(ksize[1] / 2)
    xmax = np.ceil(max(1, xmax))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gabor_triangle = np.zeros([ksize[0], ksize[1]])
    #    gabor_triangle = np.zeros([2*ksize[0]+1, 2*ksize[1]+1])
    if cos_or_sin == 1:
        for j in range(0, 100, 1):
            gabor_triangle_tmp = 4 / math.pi * (1 / (2 * j + 1)) * np.cos(
                2 * np.pi / Lambda * ((2 * j + 1)) * x_theta + j * math.pi)
            gabor_triangle = gabor_triangle + gabor_triangle_tmp
    if cos_or_sin == 0:
        for j in range(0, 100, 1):
            gabor_triangle_tmp = 4 / math.pi * (1 / (2 * j + 1)) * np.sin(
                2 * np.pi / Lambda * ((2 * j + 1)) * x_theta + j * math.pi)
            gabor_triangle = gabor_triangle + gabor_triangle_tmp
    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * gabor_triangle
    return gb


def createGabor(in_chan, out_chan):
    sigm = 3
    lambd = 2 * sigm
    gamm = 1
    k_size = (3, 3)
    direction = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]

    # ablation
    # direction = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120,
    #              125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175]
    # direction = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    # direction = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]

    gabor_filter = nn.ModuleList()

    scaling_function = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ]).astype(np.float32)

    for i, t in enumerate(direction):
        # real part
        gabor_kernel_real = gabor_gen(sigma=sigm, theta=t * math.pi / 180 * 2, Lambda=lambd, gamma=gamm, ksize=k_size,
                                      cos_or_sin=1)
        new_gabor_kernel = np.where(gabor_kernel_real > 0, gabor_kernel_real, 0)

        count1 = np.sum(new_gabor_kernel)
        gabor_kernel_real = (gabor_kernel_real / count1).astype(np.float32) * scaling_function

        filter = gabor_kernel_real.reshape((1, 1, 3, 3))
        filter = np.repeat(filter, in_chan, axis=1)
        filter = np.repeat(filter, out_chan, axis=0)

        filter = torch.from_numpy(filter)
        filter = nn.Parameter(filter, requires_grad=False)
        conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
        conv.weight = filter
        gabor_real = nn.Sequential(conv, nn.BatchNorm2d(out_chan))

        gabor_filter.append(gabor_real)

        # imagine part
        gabor_kernel_imagine = gabor_gen(sigma=sigm, theta=t * math.pi / 180 * 2, Lambda=lambd, gamma=gamm,
                                         ksize=k_size,
                                         cos_or_sin=0)
        new_gabor_kernel = np.where(gabor_kernel_imagine > 0, gabor_kernel_imagine, 0)

        count1 = np.sum(new_gabor_kernel)
        gabor_kernel_imagine = (gabor_kernel_imagine / count1).astype(np.float32) * scaling_function

        filter = gabor_kernel_imagine.reshape((1, 1, 3, 3))
        filter = np.repeat(filter, in_chan, axis=1)
        filter = np.repeat(filter, out_chan, axis=0)

        filter = torch.from_numpy(filter)
        filter = nn.Parameter(filter, requires_grad=False)
        conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
        conv.weight = filter
        gabor_imagine = nn.Sequential(conv, nn.BatchNorm2d(out_chan))

        gabor_filter.append(gabor_imagine)

        # result = cv2.filter2D(image, -1, gabor_kernel)
        # out += result
    return gabor_filter


def run_gabor(operators, feat):
    B, C, H, W = feat.shape
    out = torch.zeros([B, 1, H, W]).cuda()
    for gabor in operators:
        result = gabor(feat)
        out += result

    out = out / (len(operators) / 2)
    return torch.sigmoid(out) * feat


class Boundary_Detector(nn.Module):
    def __init__(self, in_dim1, in_dim2):
        super(Boundary_Detector, self).__init__()
        # self.eam = EAM()
        # self.sobel_x, self.sobel_y = get_sobel(in_dim1, 1)
        # self.sobel_x2, self.sobel_y2 = get_sobel(in_dim2, 1)
        # self.laplace1 = get_laplace(in_dim1, 1)
        # self.laplace2 = get_laplace(in_dim2, 1)
        self.gabor = createGabor(in_dim1, 1)

    def forward(self, x_fuse):
        # boundary1 = run_sobel(self.sobel_x, self.sobel_y, x_fuse)
        # boundary2 = run_sobel(self.sobel_x2, self.sobel_y2, x_depth)
        # boundary1 = run_laplace(self.laplace1, x_fuse)
        # boundary2 = run_laplace(self.laplace2, x_depth)
        # boundary = torch.cat((boundary1, boundary2), dim=1)
        # boundary = run_laplace(self.laplace, x)
        # edge = self.eam(boundary1, boundary2)
        # attn = torch.sigmoid(edge)
        boundary = run_gabor(self.gabor, x_fuse)

        return boundary
