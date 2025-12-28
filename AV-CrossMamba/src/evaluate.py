import argparse
import torch
import time
from utils import *
import os
from new_model import MambaAVSepFormer_warpper
from model import Cross_Sepformer_warpper
from muse import muse
from mir_eval.separation import bss_eval_sources
from pystoi import stoi
from pesq import pesq
import torch.utils.data as data
from scipy.io import wavfile
import numpy as np
import math
import tqdm
from thop import profile  # 导入 thop 库用于计算 MACs

MAX_INT16 = np.iinfo(np.int16).max


def write_wav(fname, samps, sampling_rate=16000, normalize=True):
    """
    Write wav files in int16, support single/multi-channel
    写入 WAV 文件，支持单/多通道，并进行归一化。
    """
    # for multi-channel, accept ndarray [Nsamples, Nchannels]
    samps = np.divide(samps, np.max(np.abs(samps)))

    # same as MATLAB and kaldi
    if normalize:
        samps = samps * MAX_INT16
        samps = samps.astype(np.int16)
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)

    # 使用 scipy.io.wavfile 写入
    wavfile.write(fname, sampling_rate, samps)


def SDR(est, egs, mix):
    '''
    calculate SDRi (SDR improvement)
    计算 SDRi (SDR 改善度)
    est: Network generated audio (numpy array)
    egs: Ground Truth (numpy array)
    mix: Mixture audio (numpy array)
    '''
    # bss_eval_sources 期望形状 (n_sources, n_samples)
    sdr, _, _, _ = bss_eval_sources(np.expand_dims(egs, 0), np.expand_dims(est, 0))
    mix_sdr, _, _, _ = bss_eval_sources(np.expand_dims(egs, 0), np.expand_dims(mix, 0))
    # Return SDR improvement
    return float(sdr - mix_sdr)


class dataset(data.Dataset):
    def __init__(self,
                 mix_lst_path,
                 audio_direc,
                 visual_direc,
                 mixture_direc,
                 batch_size=4,  # 评估时通常为 1
                 partition='test',
                 sampling_rate=16000,
                 mix_no=2):
        self.minibatch = []
        self.audio_direc = audio_direc
        self.visual_direc = visual_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C = mix_no

        mix_csv = open(mix_lst_path).read().splitlines()
        # 仅加载对应 partition 的行
        self.mix_lst = list(filter(lambda x: x.split(',')[0] == partition, mix_csv))

    def __getitem__(self, index):
        line = self.mix_lst[index]
        parts = line.split(',')

        # 假设文件列表中的最后一个元素是持续时间（秒）
        length_sec = float(parts[-1])
        min_length = int(length_sec * self.sampling_rate)

        # 1. 构造混合音频路径 (Mix Split, C * [Split, ID, Path, dB], Duration)
        mix_split_name = parts[0]
        # 文件名包含所有说话人的信息部分，即 parts[1] 到 parts[-2]
        mixture_filename_parts = parts[1:-1]
        mixture_filename = '_'.join(mixture_filename_parts).replace('/', '_') + '.wav'

        mixture_path = os.path.join(self.mixture_direc, mix_split_name, mixture_filename)

        _, mixture = wavfile.read(mixture_path)
        mixture_single = self._audio_norm(mixture[:min_length])

        # === 2. 目标语音和视觉特征 (多份) ===
        audios, visuals = [], []
        SPEAKER_BLOCK_SIZE = 4

        for c in range(self.C):
            start_idx = 1 + c * SPEAKER_BLOCK_SIZE

            split_name = parts[start_idx]
            speaker_id = parts[start_idx + 1]
            utt_path_base = parts[start_idx + 2]

            # 目标语音路径
            audio_path = os.path.join(self.audio_direc, split_name, speaker_id, utt_path_base + '.wav')

            _, audio = wavfile.read(audio_path)
            audio = self._audio_norm(audio[:min_length])
            audios.append(audio)

            # 视觉特征路径
            visual_path = os.path.join(self.visual_direc, split_name, speaker_id, utt_path_base + '.npy')

            visual = np.load(visual_path)
            # 帧率 25 fps
            length = math.floor(min_length / self.sampling_rate * 25)
            visual = visual[:length, ...]
            if visual.shape[0] < length:
                visual = np.pad(visual, ((0, int(length - visual.shape[0])), (0, 0)), mode='edge')
            visuals.append(visual)

        # 将列表堆叠成 numpy 数组
        audios = np.stack(audios, axis=0)  # [C, T]
        visuals = np.stack(visuals, axis=0)  # [C, T_v, D_v]

        # === 3. 将混合语音复制 C 份 ===
        mixtures = np.stack([mixture_single] * self.C, axis=0)  # [C, T]

        # 修复逻辑: 返回第一个说话人的 ID/Path 组合作为文件名标识
        # parts[2] = S1_ID, parts[3] = S1_Path
        # 这将是 'id01018/VNPt_GB5QiE/00069'
        fname_identifier = parts[2] + '/' + parts[3]

        # DataLoader 期望返回一个包含所有元素的元组，因此将字符串用逗号包裹
        # 返回结果是 (mixtures, audios, visuals, ('id/path',))
        return mixtures, audios, visuals, (fname_identifier,)

    def __len__(self):
        return len(self.mix_lst)

    def _audio_norm(self, audio):
        # 音频归一化
        max_abs = np.max(np.abs(audio))
        if max_abs == 0:
            return audio
        return np.divide(audio, max_abs)


def calculate_macs_with_thop(model, a_mix_example, v_tgt_example):
    """
    计算模型的 MACs (Multiply-Accumulate Operations) 使用 thop 库。
    """
    try:
        # profile expects inputs to be a tuple.
        macs, params = profile(model, inputs=(a_mix_example.to(next(model.parameters()).device),
                                              v_tgt_example.to(next(model.parameters()).device)),
                               verbose=False)
        # MACs / 1e9 to convert to Giga MACs
        return macs / 1e9
    except Exception as e:
        print(f"Error during MACs calculation with thop: {e}")
        return 0.0


def main(args):
    # Model
    # 根据你的配置选择模型
    model = MambaAVSepFormer_warpper()
    # model = Cross_Sepformer_warpper()
    # model = muse(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
    #                 args.C, 800)
    model = model.cuda()

    ckpt_path = os.path.join(args.continue_from, "model_dict.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pretrained_state = ckpt["model"]
    new_state = {}
    for k, v in pretrained_state.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_key] = v
    model.load_state_dict(new_state)

    datasets = dataset(
        mix_lst_path=args.mix_lst_path,
        audio_direc=args.audio_direc,
        visual_direc=args.visual_direc,
        mixture_direc=args.mixture_direc,
        mix_no=args.C)

    test_generator = data.DataLoader(datasets,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                     pin_memory=True)

    model.eval()
    if args.save_est_dir:
        os.makedirs(args.save_est_dir, exist_ok=True)

    # ====== 初始化性能度量变量 ======
    total_cpu_time = 0.0  # 秒
    total_gpu_time = 0.0  # 毫秒
    max_gpu_memory = 0.0  # MB
    total_macs_g = 0.0  # Giga MACs (只计算一次)

    is_cuda_available = torch.cuda.is_available()

    if is_cuda_available:
        torch.cuda.reset_peak_memory_stats()
        # 预热 GPU
        dummy_c = args.C
        dummy_len_s = 3
        dummy_input_a = torch.randn(dummy_c, 16000 * dummy_len_s).cuda().float()
        dummy_input_v = torch.randn(dummy_c, dummy_len_s * 25, 512).cuda().float()
        for _ in range(5):
            _ = model(dummy_input_a, dummy_input_v)
        torch.cuda.synchronize()

    with torch.no_grad():
        avg_sisnri, avg_sdri, avg_pesq, avg_stoi = 0, 0, 0, 0

        # 接收 a_mix (C份), a_tgt (C份), v_tgt (C份), fname (list/tuple)
        for i, (a_mix, a_tgt, v_tgt, fname) in enumerate(tqdm.tqdm(test_generator)):

            # --- 核心修复点：鲁棒地提取文件名字符串 ---
            # 解决 'list' object has no attribute 'replace' 的问题
            fname_str = None

            # 1. 确保 fname 至少是 list/tuple 并且不为空
            if not isinstance(fname, (list, tuple)) or len(fname) == 0:
                print(f"Warning: Unexpected fname format at index {i}. Skipping sample.")
                continue

            # 2. 提取第一个元素 (通常是 list/tuple 包装后的结果)
            temp_fname = fname[0]

            # 3. 如果第一个元素仍然是 tuple 或 list，则再向下提取一次 (解决 DataLoader 双重包装)
            if isinstance(temp_fname, (list, tuple)) and len(temp_fname) > 0:
                temp_fname = temp_fname[0]

            # 4. 最终检查是否为字符串
            if isinstance(temp_fname, str):
                fname_str = temp_fname
            else:
                print(
                    f"Warning: Failed to extract string filename at index {i}. Skipping sample. Type: {type(temp_fname)}")
                continue

            # =========================================================

            # 去除 DataLoader 的 Batch=1 维度
            a_mix = a_mix.cuda().squeeze(0).float()  # [C, T] - C 个混合音频副本
            a_tgt = a_tgt.cuda().squeeze(0).float()  # [C, T] - C 个目标音频
            v_tgt = v_tgt.cuda().squeeze(0).float()  # [C, T_v, D_v] - C 个视觉特征

            # ====== MACs 计算 (只在第一个样本上计算一次) ======
            if i == 0:
                total_macs_g = calculate_macs_with_thop(model, a_mix, v_tgt)

            # ====== GPU/CPU 时间测量 ======

            # GPU Time Setup
            if is_cuda_available:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                # CPU Time Setup
            cpu_start_time = time.time()

            # 核心推理 (Forward Pass)
            est_sources = model(a_mix, v_tgt)  # est_sources 期望形状为 [C, T]

            # CPU Time End
            cpu_end_time = time.time()
            current_cpu_time = cpu_end_time - cpu_start_time
            total_cpu_time += current_cpu_time

            # GPU Time End and Measurement
            if is_cuda_available:
                end_event.record()
                torch.cuda.synchronize()  # 等待所有 CUDA 操作完成
                current_gpu_time_ms = start_event.elapsed_time(end_event)  # 毫秒
                total_gpu_time += current_gpu_time_ms

                # GPU Memory: 检查当前峰值，并更新全局最大值
                current_peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                if current_peak_memory > max_gpu_memory:
                    max_gpu_memory = current_peak_memory

            # ============================================
            # ====== 计算所有 C 个目标的平均指标 ======
            # ============================================

            # a_mix_single: 混合音频只取第一个副本 [1, T] - 用于计算 SI-SNRi / SDRi
            a_mix_single = a_mix[0].unsqueeze(0)
            mix_np = a_mix_single.squeeze().cpu().numpy()  # 原始混合音频的 numpy 版本

            current_sisnri = 0
            current_sdri = 0
            current_pesq = 0
            current_stoi = 0

            # 遍历 C 个目标，计算平均指标
            for c in range(args.C):
                a_tgt_c = a_tgt[c].unsqueeze(0)
                est_c = est_sources[c].unsqueeze(0)

                # 1. SI-SNRi Calculation
                # 确保 cal_SISNR 已正确导入 (假定它在 utils.py 中)
                sisnr_mix = cal_SISNR(a_tgt_c, a_mix_single)
                sisnr_est = cal_SISNR(a_tgt_c, est_c)
                current_sisnri += (sisnr_est - sisnr_mix).mean().item()

                # 2. SDRi / PESQ / STOI Calculation (需要 NumPy 数组)
                est_np_c = est_c.squeeze().cpu().numpy()
                tgt_np_c = a_tgt_c.squeeze().cpu().numpy()

                # SDRi:
                current_sdri += SDR(est_np_c, tgt_np_c, mix_np)

                # PESQ: 宽带 (wb)
                try:
                    current_pesq += pesq(16000, tgt_np_c, est_np_c, 'wb')
                except Exception:
                    # PESQ 可能会因为长度问题偶尔失败，设置为 0.0 避免崩溃
                    current_pesq += 0.0

                # STOI:
                current_stoi += stoi(tgt_np_c, est_np_c, 16000, extended=False)

            # 将 C 个目标指标的平均值（当前样本的平均得分）加入总平均
            avg_sisnri += current_sisnri / args.C
            avg_sdri += current_sdri / args.C
            avg_pesq += current_pesq / args.C
            avg_stoi += current_stoi / args.C

            # ==================================
            # ====== 音频保存（保存所有 C 个分离结果）======
            # ==================================

            # 只保存前 3 个样本的音频结果
            if args.save_est_dir and i < 3 and fname_str:
                # 使用提取出的字符串进行 replace，现在 fname_str 保证是字符串
                save_dir = os.path.join(args.save_est_dir, fname_str.replace('/', '_'))
                os.makedirs(save_dir, exist_ok=True)

                # 遍历并保存 C 个分离出的语音
                for c in range(args.C):
                    est_spk_np = est_sources[c].squeeze().cpu().numpy()
                    est_spk_path = os.path.join(save_dir, f"est_spk{c + 1}.wav")
                    write_wav(est_spk_path, est_spk_np, sampling_rate=16000)

                    # 对应的第 c 个目标语音 (Ground Truth)
                    tgt_spk_np = a_tgt[c].squeeze().cpu().numpy()
                    tgt_spk_path = os.path.join(save_dir, f"tgt_spk{c + 1}.wav")
                    write_wav(tgt_spk_path, tgt_spk_np, sampling_rate=16000)

                # 保存混合音频
                mix_path = os.path.join(save_dir, "mixture.wav")
                write_wav(mix_path, mix_np, 16000)

        # ====== 结果展示 ======
        total = i + 1

        # 性能指标平均
        avg_sisnri /= total
        avg_sdri /= total
        avg_pesq /= total
        avg_stoi /= total

        # 计算平均时间
        avg_cpu_time_s = total_cpu_time / total
        avg_gpu_time_ms = total_gpu_time / total

        print("\n=== Evaluation Results (Average over all C targets) ===")
        print(f"Avg SI-SNRi: {avg_sisnri:.3f} dB")
        print(f"Avg SDRi:    {avg_sdri:.3f} dB")
        print(f"Avg PESQ:    {avg_pesq:.3f}")
        print(f"Avg STOI:    {avg_stoi:.3f}")

        print("\n=== Computational Costs (Per Sample) ===")
        print(f"MACs (G):           {total_macs_g:.2f} G")
        print(f"Avg Test Time CPU:  {avg_cpu_time_s:.4f} s")
        if is_cuda_available:
            print(f"Avg Test Time GPU:  {avg_gpu_time_ms:.2f} ms")
            print(f"Peak GPU Memory:    {max_gpu_memory:.2f} MB")
        else:
            print("GPU metrics are not available (CUDA not found).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataloader
    parser.add_argument('--mix_lst_path', type=str,
                        default='/home/wangcanyang/MuSE/data/voxceleb2-800/mixture_data_list_2mix.csv',
                        help='directory including train data')
    parser.add_argument('--audio_direc', type=str, default='/home/wangcanyang/vox2clean_audio/',
                        help='directory including validation data')
    parser.add_argument('--visual_direc', type=str, default='/home/wangcanyang/vox2lip/',
                        help='directory including test data')
    parser.add_argument('--mixture_direc', type=str, default='/home/wangcanyang/vox2_2mix/2_mix/',
                        help='directory of audio')

    parser.add_argument('--continue_from', type=str,
                        default='/home/wangcanyang/MuSE/src/logs/avaNet_2025-10-03(10:13:39)/')

    # Training
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers to generate minibatch')

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
    parser.add_argument('--save_est_dir', type=str, default='',
                        help='Directory to save estimated samples (save first 3 only)')
    args = parser.parse_args()

    # 导入 utils 模块（假定其中包含 cal_SISNR）
    try:
        from utils import cal_SISNR
    except ImportError:
        # 如果 utils 导入失败，提供一个存根函数避免崩溃
        print("Warning: Failed to import cal_SISNR from utils. Using a dummy function.")


        def cal_SISNR(a, b):
            # 返回一个 dummy tensor
            return torch.tensor([0.0]).cuda()

    main(args)