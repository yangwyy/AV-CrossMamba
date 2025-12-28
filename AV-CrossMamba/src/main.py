import argparse
import torch
from utils import *
import os
# from model import Cross_Sepformer_warpper
# from muse import muse
from new_model import MambaAVSepFormer_warpper
from solver import Solver


def main(args):
    if args.distributed:
        torch.manual_seed(0)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # speaker id assignment
    mix_lst=open(args.mix_lst_path).read().splitlines()
    train_lst=list(filter(lambda x: x.split(',')[0]=='train', mix_lst))
    IDs = 0
    speaker_dict={}
    for line in train_lst:
        parts = line.split(',')  # 按逗号分割当前样本的所有字段
        # 循环范围为 args.C（混合的说话人数量）
        for i in range(args.C):
            # 计算第i个说话人ID的索引：
            # 每个说话人占4个字段，第1个说话人ID在索引2，因此公式为：1 + i*4 + 1 = i*4 + 2
            id_index = i * 4 + 2  # 推导：i=0→2，i=1→6，i=2→10...与样本匹配

            # 防止索引越界（避免样本格式错误导致崩溃）
            if id_index >= len(parts):
                print(f"Warning: 样本格式错误，字段不足 - 行内容：{line}")
                continue

            speaker_id = parts[id_index]  # 获取第i个说话人的ID
            if speaker_id not in speaker_dict:
                speaker_dict[speaker_id] = IDs  # 分配唯一整数ID
                IDs += 1

    # 将说话人映射表和总数存入参数
    args.speaker_dict = speaker_dict
    args.speakers = len(speaker_dict)

    # Model
    # model = MambaAVSepFormer_warpper(
    #     kernel_size=16,
    #     N_encoder_out=256,
    #     num_spks=1,
    #     n_mamba=8,
    #     d_state=16,
    #     expand=2,
    #     d_conv=4,
    #     bidirectional=False,
    #     fused_add_norm=True,
    #     rms_norm=True,
    #     norm_epsilon=1e-5
    # )
    # model = Cross_Sepformer_warpper()
    # model = muse(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
    #                 args.C, 800)
    model = MambaAVSepFormer_warpper()
    if (args.distributed and args.local_rank ==0) or args.distributed == False:
        print("started on " + args.log_name + '\n')
        print(args)
        print("\nTotal number of parameters: {} \n".format(sum(p.numel() for p in model.parameters())))
        print(model)

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_sampler, train_generator = get_dataloader(args,'train')
    _, val_generator = get_dataloader(args, 'val')
    _, test_generator = get_dataloader(args, 'test')
    args.train_sampler=train_sampler

    solver = Solver(args=args,
                model = model,
                optimizer = optimizer,
                train_data = train_generator,
                validation_data = val_generator,
                test_data = test_generator)
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("avConv-tasnet")

    # Dataloader
    parser.add_argument('--mix_lst_path', type=str, default='/home/panzexu/datasets/LRS2/audio/2_mix_min/mixture_data_list_2mix.csv',
                        help='directory including train data')
    parser.add_argument('--audio_direc', type=str, default='/home/panzexu/datasets/LRS2/audio/Audio/',
                        help='directory including validation data')
    parser.add_argument('--visual_direc', type=str, default='/home/panzexu/datasets/LRS2/lip/',
                        help='directory including test data')
    parser.add_argument('--mixture_direc', type=str, default='/home/panzexu/datasets/LRS2/audio/2_mix_min/',
                        help='directory of audio')

    # Training
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size')
    parser.add_argument('--max_length', default=6, type=int,
                        help='max_length of mixture in training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to generate minibatch')
    parser.add_argument('--epochs', default=200, type=int,
                        help='Number of maximum epochs')

    # Model hyperparameters
    # Model hyperparameters
    parser.add_argument('--L', default=40, type=int,
                        help='Length of the filters in samples (80=5ms at 16kHZ)')
    parser.add_argument('--N', default=256, type=int,
                        help='Number of filters in autoencoder')
    parser.add_argument('--B', default=256, type=int,
                        help='Number of channels in bottleneck 1 × 1-conv block')
    parser.add_argument('--C', type=int, default=2,
                        help='number of speakers to mix')
    parser.add_argument('--H', default=512, type=int,
                        help='Number of channels in convolutional blocks')
    parser.add_argument('--P', default=3, type=int,
                        help='Kernel size in convolutional blocks')
    parser.add_argument('--X', default=8, type=int,
                        help='Number of convolutional blocks in each repeat')
    parser.add_argument('--R', default=4, type=int,
                        help='Number of repeats')

    # optimizer
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Init learning rate')
    parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')


    # Log and Visulization
    parser.add_argument('--log_name', type=str, default=None,
                        help='the name of the log')
    parser.add_argument('--use_tensorboard', type=int, default=0,
                        help='Whether to use use_tensorboard')
    parser.add_argument('--continue_from', type=str, default='',
                        help='Whether to use use_tensorboard')

    # Distributed training
    parser.add_argument('--opt-level', default='O0', type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--patch_torch_functions', type=str, default=None)

    args = parser.parse_args()

    args.distributed = False
    args.world_size = 1
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    main(args)
