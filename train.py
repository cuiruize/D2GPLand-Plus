import os
import argparse
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from src.prepare_dataset import prepare_dataset
from src.dataset import RoboticsDataset, save_img
from models.D2GPLandPlus import ESANetOneModality_sam as D2GPLandPlus
from pytorch_metric_learning import losses
from medpy.metric.binary import assd


class Edge_Prototypes(nn.Module):
    def __init__(self, num_classes=3, feat_dim=256):
        super(Edge_Prototypes, self).__init__()
        self.class_embeddings = nn.Embedding(num_classes, feat_dim)

    def forward(self):
        return self.class_embeddings.weight


def _dice_loss(pred, target):
    smooth = 1e-5
    pred = F.softmax(pred, dim=1)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3))
    dice = 1 - ((2 * inter + smooth) / (union + smooth))
    return dice.mean()


class BBCEWithLogitLoss(nn.Module):
    '''Balanced BCEWithLogitLoss'''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)
        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)
        return loss


def MIoU(pred, gt):
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    dice = (2 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
    iou = dice / (2 - dice)
    return iou, dice


def train(args):
    train_file, test_file, val_file = prepare_dataset.get_split(args.data_path)
    
    train_dataset = RoboticsDataset(train_file, transform=None, mode='train')
    test_dataset = RoboticsDataset(test_file, transform=None, mode='test')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    
    best_dice = -100
    best_iou = -100
    best_assd = -100

    model = D2GPLandPlus(args.img_size, args.img_size).to(device)
    edge_prototypes_model = Edge_Prototypes(
        num_classes=args.num_classes, 
        feat_dim=args.feat_dim
    ).to(device)

    # freeze depth anything and sam
    model.depth_encoder.requires_grad_(False)
    model.sam_encoder.requires_grad_(False)
    model.prompt_fusion_layer.boundary_detectors.requires_grad_(False)

    # contrastive loss
    contrastive_loss_model = losses.NTXentLoss(temperature=args.temperature).cuda()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': edge_prototypes_model.parameters()}
    ], lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, args.min_lr
    )

    for epoch in range(args.epochs):
        print('lr: ' + str(optimizer.param_groups[0]['lr']))
        epoch_running_loss = 0
        epoch_seg_loss = 0
        epoch_contrastive_loss = 0
        epoch_edge_loss = 0

        for batch_idx, (X_batch, depth, y_batch, *rest) in tqdm(enumerate(train_loader)):
            model.train()
            edge_prototypes_model.train()

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            depth = depth.to(device)
            prototypes = edge_prototypes_model()

            output, feature, edge_out, _, _ = model(X_batch, prototypes, depth)

            class_embs = None
            for cls_id in range(prototypes.shape[0]):
                mask = y_batch[:, cls_id + 1, :, :]
                mask = torch.stack([mask for _ in range(feature.shape[1])], dim=1)
                mask = F.interpolate(mask, size=feature.shape[2], mode="bilinear")
                class_emb = feature * mask
                class_emb = F.interpolate(class_emb, size=16, mode="bilinear")
                class_emb = class_emb.mean(1).reshape(-1, 16 * 16)

                if class_embs is None:
                    class_embs = class_emb
                else:
                    class_embs = torch.cat((class_embs, class_emb), dim=0)

            prototype_loss = contrastive_loss_model(
                prototypes,
                torch.tensor([i for i in range(1, (prototypes.size()[0] + 1))]).to(device),
                ref_emb=class_embs,
                ref_labels=torch.tensor([1, 1, 2, 2, 3, 3,])
            )

            seg_loss = _dice_loss(output, y_batch) + bce_loss(output, y_batch)
            edge_gt = F.interpolate(y_batch[:, 1:, :, :], size=16, mode="bilinear")
            edge_loss = _dice_loss(edge_out, edge_gt) + bce_loss(edge_out, edge_gt)
            loss = seg_loss + prototype_loss + edge_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_running_loss += loss.item()
            epoch_seg_loss += seg_loss.item()
            epoch_contrastive_loss += prototype_loss.item()
            epoch_edge_loss += edge_loss.item()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch, args.epochs, epoch_running_loss / (batch_idx + 1)))
        print('epoch [{}/{}], seg loss:{:.4f}'.format(epoch, args.epochs, epoch_seg_loss / (batch_idx + 1)))
        print('epoch [{}/{}], contrastive loss:{:.4f}'.format(epoch, args.epochs, epoch_contrastive_loss / (batch_idx + 1)))
        print('epoch [{}/{}], edge loss:{:.4f}'.format(epoch, args.epochs, epoch_edge_loss / (batch_idx + 1)))

        if (epoch % args.val_interval) == 0:
            model.eval()
            edge_prototypes_model.eval()

            validation_IOU = []
            mDice = []
            mAssd = []

            with torch.no_grad():
                for X_batch, depth, y_batch, name in tqdm(test_loader):
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    depth = depth.to(device)
                    prototypes = edge_prototypes_model()

                    y_out, _, _, out_cnn, task_token = model(X_batch, prototypes, depth)
                    y_out = torch.argmax(torch.softmax(y_out, dim=1), dim=1)
                    y_batch = torch.argmax(y_batch, dim=1)

                    tmp2 = y_batch.detach().cpu().numpy()
                    tmp = y_out.detach().cpu().numpy()
                    tmp = tmp[0]
                    tmp2 = tmp2[0]

                    pred = np.array([tmp == i for i in range(4)])
                    gt = np.array([tmp2 == i for i in range(4)])

                    iou, dice = MIoU(pred[1:].astype(np.uint8).flatten(), gt[1:].astype(np.uint8).flatten())

                    if 0 == np.count_nonzero(pred[1:]) or 0 == np.count_nonzero(gt[1:]):
                        mAssd.append(80)
                    else:
                        assd_value = assd(pred[1:], gt[1:])
                        mAssd.append(assd_value)

                    validation_IOU.append(iou)
                    mDice.append(dice)

            print(np.mean(validation_IOU))
            print(np.mean(mDice))
            
            if np.mean(mDice) > best_dice:
                best_dice = np.mean(mDice)
                best_iou = np.mean(validation_IOU)
                best_assd = np.mean(mAssd)
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_path.pth"))
                torch.save(edge_prototypes_model.state_dict(), os.path.join(args.save_dir, "best_prototype.pth"))
            
            print("best dice is:{:.4f}".format(best_dice))
            print("best iou is:{:.4f}".format(best_iou))
            print("best assd is:{:.4f}".format(best_assd))
        
        scheduler.step()

def parse_args():
    parser = argparse.ArgumentParser(description='D2GPLandPlus Training')
    
    parser.add_argument('--data_path', type=str, default='L3D-2K/', help='dataset name or path')

    parser.add_argument('--num_workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--val_interval', type=int, default=1, help='validation interval (epochs)')
    
    parser.add_argument('--img_size', type=int, default=1024, help='input image size')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes (excluding background)')
    parser.add_argument('--feat_dim', type=int, default=256, help='feature dimension')
    parser.add_argument('--temperature', type=float, default=0.07, help='contrastive learning temperature')
    
    parser.add_argument('--device', type=str, default='cuda', help='training device')
    parser.add_argument('--save_dir', type=str, default='d2gpland+_results', help='model save directory')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("=" * 50)
    print("Training Configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 50)
    
    best_dice, best_iou, best_assd = train(args)
    
    print("\n" + "=" * 50)
    print("Training Completed!")
    print(f"  Best Dice: {best_dice:.4f}")
    print(f"  Best IoU: {best_iou:.4f}")
    print(f"  Best ASSD: {best_assd:.4f}")
    print("=" * 50)
