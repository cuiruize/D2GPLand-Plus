import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import csv
import yaml
import cv2 as cv
from medpy.metric import assd
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from src.prepare_dataset import prepare_dataset
from src.dataset import RoboticsDataset, save_img
from models.D2GPLandPlus import ESANetOneModality_sam


class Edge_Prototypes(nn.Module):
    def __init__(self, num_classes=3, feat_dim=256):
        super(Edge_Prototypes, self).__init__()
        self.class_embeddings = nn.Embedding(num_classes, feat_dim)

    def forward(self):
        return self.class_embeddings.weight


def evaluation(pred, gt):
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    dice = (2 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
    iou = dice / (2 - dice)

    return iou, dice


def main(save_path, args):
    train_file, test_file, val_file = prepare_dataset.get_split(args.data_path)

    test_dataset = RoboticsDataset(test_file, transform=None, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda")

    model = ESANetOneModality_sam(1024, 1024).to(device)
    model_checkpoint = torch.load(args.model_path)
    model.load_state_dict(model_checkpoint)
    edge_prototypes_model = Edge_Prototypes(num_classes=3, feat_dim=256).to(device)
    prototype_checkpoint = torch.load(args.prototype_path)
    edge_prototypes_model.load_state_dict(prototype_checkpoint)

    path = torch.load(args.model_path)
    model.load_state_dict(path)
    model.eval()
    edge_prototypes_model.eval()
    model.to(device)
    edge_prototypes_model.to(device)

    validation_IOU = []
    mDice = []
    mAssd = []

    for index, (X_batch, depth, y_batch, name) in tqdm(enumerate(test_loader)):

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        depth = depth.to(device)
        prototypes = edge_prototypes_model()

        output, _, _, _, _ = model(X_batch, prototypes, depth)

        output = torch.argmax(torch.softmax(output, dim=1), dim=1)
        y_batch = torch.argmax(y_batch, dim=1)
        tmp2 = y_batch.detach().cpu().numpy()
        tmp = output.detach().cpu().numpy()
        tmp = tmp[0]
        tmp2 = tmp2[0]

        pred = np.array([tmp == i for i in range(4)]).astype(np.uint8)
        gt = np.array([tmp2 == i for i in range(4)]).astype(np.uint8)

        iou, dice = evaluation(pred[1:].flatten(), gt[1:].flatten())

        if 0 == np.count_nonzero(pred[1:]):
            mAssd.append(80)
        else:
            assd_value = assd(pred[1:], gt[1:])
            mAssd.append(assd_value)

        validation_IOU.append(iou)
        mDice.append(dice)

        toprint = save_img(tmp)
        cv2.imwrite(save_path + str(name).split('/', 6)[-1].replace('/', '_')[:-3], toprint)
        

    print(np.mean(validation_IOU))
    print(np.mean(mDice))
    print(np.mean(mAssd))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        default="PATH_TO_MODEL/best_path.pth")
    parser.add_argument('--prototype_path',
                        default="PATH_TO_PROTOTYPE/best_prototype.pth")
    parser.add_argument('--data_path', default="L3D-2K/")
    args = parser.parse_args()

    save_path = "results/"
    os.makedirs(save_path, exist_ok=True)

    main(save_path, args=args)
