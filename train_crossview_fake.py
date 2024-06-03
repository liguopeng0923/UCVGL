import os
import time
import copy
import yaml
import random
import logging
_logger = logging.getLogger('train')
import argparse
import builtins
import warnings

import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from timm.utils import setup_default_logging
from datetime import datetime
from dataset.CVACT import CVACTTrainFake,CVACTVal
from dataset.CVUSA import CVUSATrainFake,CVUSAVal

from eval.evaluate import evaluate
from model.sample4geo import Sample4Geo
from model.infonce import InfoNCELoss
from utils import *
        

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=28, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--eval-batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', default='./result/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10000', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--data-folder', default='./data/CVACT', type=str, metavar='PATH',
                    help='path to dataset')
parser.add_argument('--dataset', default='cvact', type=str,
                    help='vigor, cvusa, cvact')
parser.add_argument('--op', default='adamw', type=str)
parser.add_argument('--grd-size',type=int, nargs='+', default=[384, 384],help="the size of ground images")
parser.add_argument('--sat-size',type=int, nargs='+', default=[384, 384],help="the size of satellite images")
parser.add_argument('--mean',type=int, nargs='+', default=[0.485, 0.456, 0.406],help="the mean of normalized images")
parser.add_argument('--std',type=int, nargs='+', default=[0.229, 0.224, 0.225],help="the std of normalized images")
parser.add_argument('--eval-freq',default=4, type=int,help="the frequency of evaluation")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main():
    args = parser.parse_args()
    print(args)
    args_dict = vars(args)
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    else:
        timestamp = time.time()
        local_time = time.localtime(timestamp)
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S", local_time)
        args.save_path = os.path.join(args.save_path,args.dataset,'crossview_fake',time_str)
        os.makedirs(args.save_path)
        
    with open(os.path.join(args.save_path,"args.yaml"), 'w') as file:
        yaml.dump(args_dict, file)

        
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    is_best = False
    best_acc = 0.
    
    setup_default_logging(log_path=f'{args.save_path}/train.log')
    
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node
    
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    set_up_system(args)
    
    # create model
    end = time.time()
    if args.gpu==0:
        _logger.info("=> creating model")
    
    if not args.multiprocessing_distributed or (dist.is_initialized() and args.gpu == 0):
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
    
        
    model = Sample4Geo(args)
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
            
    elif args.gpu is not None:
        model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    # compute_complexity(model, args)  # uncomment to see detailed computation cost
    criterion = InfoNCELoss(args).cuda(args.gpu)
    parameters.extend(criterion.parameters())
    
    optimizer = torch.optim.AdamW(parameters, args.lr)
    
    # optionally checkpoint from a checkpoint
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            if args.gpu==0:
                _logger.info("=> loading checkpoint '{}'".format(args.checkpoint))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            
            model.load_state_dict(checkpoint['model'])
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            if args.gpu==0:
                _logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
        else:
            if args.gpu==0:
                _logger.info("=> no checkpoint found at '{}'".format(args.checkpoint))

    if args.gpu==0:
        _logger.info(f"=> creating model cost '{time.time()-end}'")
    end = time.time()
    
    if args.gpu==0:
        _logger.info("=> creating dataset")
    
    if args.dataset.lower() == "cvusa":
        val_query_dataset = CVUSAVal(args)
    elif args.dataset.lower() == "cvact":
        val_query_dataset = CVACTVal(args)
    elif args.dataset.lower() == "vigor":
        val_query_dataset = CVACTVal(args)
    else:
        print('not implemented!')
        raise Exception
    
    val_reference_dataset = copy.deepcopy(val_query_dataset)
    val_reference_dataset.img_type = "sat"

    val_query_loader = torch.utils.data.DataLoader(
        val_query_dataset,batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True) # 512, 64
    val_reference_loader = torch.utils.data.DataLoader(
        val_reference_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True) # 80, 128
    
    if args.evaluate:
        if not args.multiprocessing_distributed or args.gpu == 0:
            evaluate(args, model, val_reference_loader,val_query_loader)
        return
    
    if args.dataset.lower() == "cvusa":
        train_dataset = CVUSATrainFake(args)
    elif args.dataset.lower() == "cvact":
        train_dataset = CVACTTrainFake(args)
    elif args.dataset.lower() == "vigor":
        train_dataset = CVACTTrainFake(args)
    else:
        print('not implemented!')
        raise Exception
    
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=train_sampler is None,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    if args.gpu==0:
        _logger.info(f"=> creating dataset cost {time.time() - end}")

    if args.gpu==0:
        _logger.info("cross-view self-supervised with fake images")
        
    for epoch in range(args.start_epoch,args.epochs):
        if args.gpu==0:
            _logger.info('start epoch:{}, date:{}'.format(epoch, datetime.now()))
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        
        lr = adjust_learning_rate(optimizer, epoch,lr=args.lr,total_epoch=args.epochs)
        
        if args.gpu == 0:
            _logger.info(f"The learning rate of epoch {epoch} is {lr}")
        
        train(train_loader, model, criterion, optimizer, epoch, args)
                
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            if  (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs: 
                result = evaluate(args, model, val_reference_loader,val_query_loader)
                _logger.info(f"=========================Recall==========================\n {result}")
                # remember best acc@1 and save checkpoint
                is_best = result[0] > best_acc
                best_acc = max(result[0], best_acc)
        
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.gpu % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best, filename=f'checkpoint.pth.tar', save_path=args.save_path)
                
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.gpu % ngpus_per_node == 0):        
        current_file_path = os.path.abspath(__file__)
        current_file_dir = os.path.dirname(current_file_path)
        ckpt_path = f"{current_file_dir}/ckpt/{args.dataset.lower()}"

        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)  
        # save_checkpoint({
        #                 'epoch': args.epochs,
        #                 'model': model.state_dict(),
        #                 'best_acc': best_acc,
        #                 'optimizer': optimizer.state_dict(),
        #             }, is_best=False, filename=f'train_crossviewfake.pth.tar', save_path=ckpt_path)
        
        shutil.copyfile(os.path.join(args.save_path,'model_best.pth.tar'), ckpt_path)
        



def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        args,
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    
    for i, (grd,fake) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            grd = grd.cuda(args.gpu, non_blocking=True)
            fake = fake.cuda(args.gpu, non_blocking=True)
            
            
        embed_q, embed_k = model(im_q = grd, im_k=fake)
        
        loss = criterion(embed_q, embed_k,label_smoothing=0.3)
            
        
        losses.update(loss.item(), grd.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


class ProgressMeter(object):
    def __init__(self, args,num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.args = args

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.args.gpu==0:
            _logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':
    main()