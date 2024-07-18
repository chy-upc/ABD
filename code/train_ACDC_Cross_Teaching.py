import argparse
import logging
import os
import random
import shutil
import sys
import time
import numpy as np
import torch
import gc
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, WeakStrongAugment)
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.unet import UNet
from networks.config import get_config
from utils import losses, ramps
from val_2D import test_single_volume
from utils.displacement import ABD_I, ABD_R

gc.collect()
torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/chy_data/ABD-main/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='train_ACDC_Cross_Teaching', help='experiment_name')
parser.add_argument('--model_1', type=str,
                    default='unet', help='model1_name')
parser.add_argument('--model_2', type=str,
                    default='swin_unet', help='model2_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--image_size', type=list, default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
parser.add_argument('--cfg', type=str,
                    default="/data/chy_data/ABD-main/code/configs/swin_tiny_patch4_window7_224_lite.yaml",
                    help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ",
                    default=None, nargs='+', )
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, ''full: cache all data, ''part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
# patch size
parser.add_argument('--patch_size', type=int, default=56, help='patch_size')
parser.add_argument('--h_size', type=int, default=4, help='h_size')
parser.add_argument('--w_size', type=int, default=4, help='w_size')
# top num
parser.add_argument('--top_num', type=int, default=4, help='top_num')
args = parser.parse_args()  
config = get_config(args)

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch,args.consistency_rampup)  # args.consistency=0.1 # args.consistency_rampup=200

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr  
    num_classes = args.num_classes 
    batch_size = args.batch_size  
    max_iterations = args.max_iterations 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model1 = UNet(in_chns=1, class_num=num_classes).cuda() 
    model2 = ViT_seg(config, img_size=args.image_size, num_classes=args.num_classes).cuda()  
    model2.load_from(config)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([WeakStrongAugment(args.image_size)]))  # args.image_size=[224,224]
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)  # args.labeled_num=7
    print("Train labeled {} samples".format(labeled_slice))

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)  # args.labeled_bs=8
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    model1.train()
    model2.train()
    loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1  

    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)  

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'],  sampled_batch['label']
            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()

            outputs1 = model1(volume_batch)  
            outputs1_unlabel = outputs1[args.labeled_bs:]
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs1_max = torch.max(outputs_soft1.detach(), dim=1)[0]
            pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)

            outputs2 = model2(volume_batch_strong)  
            outputs2_unlabel = outputs2[args.labeled_bs:]
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            outputs2_max = torch.max(outputs_soft2.detach(), dim=1)[0]
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False) 
            
            # ABD-I New Training Sample
            image_patch_supervised_last, label_patch_supervised_last = ABD_I(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, label_batch, label_batch_strong, args)
            image_output_supervised_1 = model1(image_patch_supervised_last.unsqueeze(1))  
            image_output_soft_supervised_1 = torch.softmax(image_output_supervised_1, dim=1)
            image_output_supervised_2 = model2(image_patch_supervised_last.unsqueeze(1))
            image_output_soft_supervised_2 = torch.softmax(image_output_supervised_2, dim=1)

            # ABD-R New Training Sample
            image_patch_last = ABD_R(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args)
            image_output_1 = model1(image_patch_last.unsqueeze(1))
            image_output_soft_1 = torch.softmax(image_output_1, dim=1)
            pseudo_image_output_1 = torch.argmax(image_output_soft_1.detach(), dim=1, keepdim=False)
            image_output_2 = model2(image_patch_last.unsqueeze(1))
            image_output_soft_2 = torch.softmax(image_output_2, dim=1)
            pseudo_image_output_2 = torch.argmax(image_output_soft_2.detach(), dim=1, keepdim=False)

            # First Step Loss
            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch_strong[:args.labeled_bs].long()) + dice_loss(outputs_soft2[:args.labeled_bs], label_batch_strong[:args.labeled_bs].unsqueeze(1)))
            pseudo_supervision1 = dice_loss(outputs_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(outputs_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1))
            # Second Step Loss
            if iter_num > 20000:
                loss3 = 0
                loss4 = 0
            else:
                loss3 = 0.5 * (ce_loss(image_output_supervised_1, label_patch_supervised_last.long()) + dice_loss(image_output_soft_supervised_1, label_patch_supervised_last.unsqueeze(1)))
                loss4 = 0.5 * (ce_loss(image_output_supervised_2, label_patch_supervised_last.long()) + dice_loss(image_output_soft_supervised_2, label_patch_supervised_last.unsqueeze(1)))
            pseudo_supervision3 = dice_loss(image_output_soft_1, pseudo_image_output_2.unsqueeze(1))
            pseudo_supervision4 = dice_loss(image_output_soft_2, pseudo_image_output_1.unsqueeze(1))
            # Total Loss
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            model1_loss = loss1 + 2 * loss3 + consistency_weight * (pseudo_supervision1 + pseudo_supervision3)
            model2_loss = loss2 + 2 * loss4 + consistency_weight * (pseudo_supervision2 + pseudo_supervision4)
            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss', model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss', model2_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))
            
            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(loader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes, patch_size=args.image_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)
                performance1 = np.mean(metric_list, axis=0)[0]
                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)
                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path, 'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model_1))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)
                logging.info('iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(loader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes, patch_size=args.image_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)
                performance2 = np.mean(metric_list, axis=0)[0]
                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)
                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path, 'model2_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model_2))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)
                logging.info('iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "/data/chy_data/ABD-main/model/Cross_Teaching/ACDC_{}_{}_labeled".format(args.exp, args.labeled_num)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    shutil.copy('/data/chy_data/ABD-main/code/train_ACDC_Cross_Teaching.py', snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
