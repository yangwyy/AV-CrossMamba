import argparse
import torch
from utils import *
import os
from model import Cross_Sepformer_warpper
from new_model import MambaAVSepFormer_warpper
from mir_eval.separation import bss_eval_sources
from pystoi import stoi
from pesq import pesq

MAX_INT16 = np.iinfo(np.int16).max


def write_wav(fname, samps, sampling_rate=16000, normalize=True):
    """
    Write wav files in int16, support single/multi-channel
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
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile instead
    wavfile.write(fname, sampling_rate, samps)


def SDR(est, egs, mix):
    '''
        calculate SDR
        est: Network generated audio
        egs: Ground Truth
    '''
    sdr, _, _, _ = bss_eval_sources(egs, est)
    mix_sdr, _, _, _ = bss_eval_sources(egs, mix)
    return float(sdr - mix_sdr)


class dataset(data.Dataset):
    def __init__(self,
                 mix_lst_path,
                 audio_direc,
                 visual_direc,
                 mixture_direc,
                 batch_size=1,
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
        self.mix_lst = list(filter(lambda x: x.split(',')[0] == partition, mix_csv))

    def __getitem__(self, index):
        line = self.mix_lst[index]

        mixture_path = self.mixture_direc + self.partition + '/' + line.replace(',', '_').replace('/', '_') + '.wav'
        _, mixture = wavfile.read(mixture_path)
        mixture = self._audio_norm(mixture)

        min_length = mixture.shape[0]

        line = line.split(',')
        c = 0
        audio_path = self.audio_direc + line[c * 4 + 1] + '/' + line[c * 4 + 2] + '/' + line[c * 4 + 3] + '.wav'
        _, audio = wavfile.read(audio_path)
        audio = self._audio_norm(audio[:min_length])

        # read visual
        visual_path = self.visual_direc + line[c * 4 + 1] + '/' + line[c * 4 + 2] + '/' + line[c * 4 + 3] + '.npy'
        visual = np.load(visual_path)
        length = math.floor(min_length / self.sampling_rate * 25)
        visual = visual[:length, ...]
        a = visual.shape[0]
        if visual.shape[0] < length:
            visual = np.pad(visual, ((0, int(length - visual.shape[0])), (0, 0)), mode='edge')

        return mixture, audio, visual, (line[c * 4 + 2] + '/' + line[c * 4 + 3])

    def __len__(self):
        return len(self.mix_lst)

    def _audio_norm(self, audio):
        return np.divide(audio, np.max(np.abs(audio)))


def main(args):
    # Model
    # model = muse(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
    #                args.C, 800)
    # model = MambaAVSepFormer_warpper()

    model = Cross_Sepformer_warpper()
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
                                     num_workers=args.num_workers)

    model.eval()
    with torch.no_grad():
        avg_sisnri = 0
        avg_sdri = 0
        avg_pesq = 0
        avg_stoi = 0
        for i, (a_mix, a_tgt, v_tgt, fname) in enumerate(tqdm.tqdm(test_generator)):
            a_mix = a_mix.cuda().squeeze().float().unsqueeze(0)
            a_tgt = a_tgt.cuda().squeeze().float().unsqueeze(0)
            v_tgt = v_tgt.cuda().squeeze().float().unsqueeze(0)

            estimate_source = model(a_mix, v_tgt)

            sisnr_mix = cal_SISNR(a_tgt, a_mix)
            sisnr_est = cal_SISNR(a_tgt, estimate_source)
            sisnri = sisnr_est - sisnr_mix
            avg_sisnri += sisnri.mean().item()
            # print(sisnri)

            est_np = estimate_source.squeeze().cpu().numpy()
            tgt_np = a_tgt.squeeze().cpu().numpy()
            mix_np = a_mix.squeeze().cpu().numpy()

            # SDRi
            avg_sdri += SDR(est_np, tgt_np, mix_np)

            # PESQ
            avg_pesq += pesq(16000, tgt_np, est_np, 'wb')

            # STOI
            avg_stoi += stoi(tgt_np, est_np, 16000, extended=False)

        # Take averages
        total = i + 1
        avg_sisnri /= total
        avg_sdri /= total
        avg_pesq /= total
        avg_stoi /= total

        print("\n=== Evaluation Results ===")
        print(f"Avg SI-SNRi: {avg_sisnri:.3f} dB")
        print(f"Avg SDRi:    {avg_sdri:.3f} dB")
        print(f"Avg PESQ:   {avg_pesq:.3f}")
        print(f"Avg STOI:   {avg_stoi:.3f}")

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
    parser.add_argument('--num_workers', default=4, type=int,
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

    args = parser.parse_args()

    main(args)