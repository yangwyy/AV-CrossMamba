import os
import numpy as np
import argparse
import tqdm
import scipy.io.wavfile as wavfile

MAX_INT16 = np.iinfo(np.int16).max


def write_wav(fname, samps, sampling_rate=16000, normalize=True):
    """
    Write wav files in int16, support single/multi-channel
    """
    if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
        samps = np.transpose(samps)
        samps = np.squeeze(samps)

    if normalize and np.max(np.abs(samps)) > 0:
        samps = samps * MAX_INT16

    samps = samps.astype(np.int16)
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    wavfile.write(fname, sampling_rate, samps)


def read_wav(fname, normalize=True):
    """
    Read wave files using scipy.io.wavfile (support multi-channel)
    """
    sampling_rate, samps_int16 = wavfile.read(fname)
    samps = samps_int16.astype(np.float64)
    if samps.ndim != 1:
        samps = np.transpose(samps)
        if samps.ndim == 2 and samps.shape[0] == 2:
            samps = samps[0, :]

    if normalize:
        samps = samps / MAX_INT16
    return sampling_rate, samps


def pad_or_truncate_audio(audio, target_samples):
    """
    Pads with zeros or truncates audio to the target_samples length.
    """
    if audio.shape[0] > target_samples:
        return audio[:target_samples]
    elif audio.shape[0] < target_samples:
        padding = np.zeros(target_samples - audio.shape[0], dtype=audio.dtype)
        return np.concatenate([audio, padding])
    return audio


def main(args):
    samples_per_second = 16000  # Assuming 16kHz sampling rate

    if args.C < 2:
        print("Error: The number of speakers (C) must be 2 or more for mixture generation.")
        return

    # Read mixture list
    try:
        with open(args.mixture_data_list, 'r') as f:
            mixture_data_list = f.read().splitlines()
    except FileNotFoundError:
        print(f"Error: Mixture data list file not found at {args.mixture_data_list}")
        return

    print(f"Total number of mixtures to generate: {len(mixture_data_list)}")

    # Expected structure: [Mix Split] + C * [Split, ID, Path, dB] + [Duration]
    SPEAKER_BLOCK_SIZE = 4
    EXPECTED_FIELDS = 1 + args.C * SPEAKER_BLOCK_SIZE + 1

    for line in tqdm.tqdm(mixture_data_list, desc=f"Generating {args.C}-speaker audio mixtures"):
        data = line.split(',')

        # 1. 字段数量检查
        if len(data) != EXPECTED_FIELDS:
            # 这行数据不完整，跳过。请确保 1_create_mixture_list.py 正确运行。
            print(
                f"Skipping line due to incorrect field count: Expected {EXPECTED_FIELDS} fields, but found {len(data)}. Line: {line}")
            continue

        # The last element is the target duration (in seconds)
        target_duration = float(data[-1])
        target_samples = int(samples_per_second * target_duration)

        # --- Mixture Path and File Name Construction ---

        # 整体混合的分割名（用于输出目录）
        mix_split_name = data[0]

        # 混合文件名：排除第一个字段（mix_split_name）和最后一个字段（target_duration）
        # 这样文件名只包含说话人的信息，更简洁。
        mixture_filename_parts = data[1:-1]
        mixture_filename = '_'.join(mixture_filename_parts).replace('/', '_') + '.wav'

        # 输出目录 based on mix_split_name
        save_direc = os.path.join(args.mixture_audio_direc, mix_split_name)
        mixture_save_path = os.path.join(save_direc, mixture_filename)

        if not os.path.exists(save_direc):
            os.makedirs(save_direc)

        if os.path.exists(mixture_save_path):
            continue

        audio_mix = None
        target_power = 0.0
        all_scaled_audios = []

        # --- Iterate over all C speakers ---
        for c in range(args.C):
            # 关键修复：起始索引必须 +1 来跳过 data[0] (mix_split_name)
            start_idx = 1 + c * SPEAKER_BLOCK_SIZE

            # 字段索引（相对于完整的 'data' 列表）
            idx_split = start_idx
            idx_id = start_idx + 1
            idx_path = start_idx + 2
            idx_db = start_idx + 3

            # 提取说话人信息
            split_name = data[idx_split]
            speaker_id = data[idx_id]
            utt_path = data[idx_path]

            # 尝试转换 db_ratio
            try:
                db_ratio = float(data[idx_db])
            except ValueError:
                # 如果转换失败，打印详细信息并跳过整个混合
                print(
                    f"CRITICAL PARSING ERROR: Speaker {c + 1} (index {idx_db}) expected float for db_ratio, got '{data[idx_db]}'. Skipping mixture: {line}")
                all_scaled_audios = []
                break

            # 构造完整路径：audio_data_direc / split_name / speaker_id / utt_path .wav
            audio_path = os.path.join(args.audio_data_direc, split_name, speaker_id, utt_path + '.wav')

            try:
                # 读取音频
                _, audio = read_wav(audio_path)

                # 填充/截断音频到固定长度
                audio = pad_or_truncate_audio(audio, target_samples)

                current_power = np.linalg.norm(audio, 2) ** 2 / audio.size

                if c == 0:
                    # 说话人 0 是目标
                    target_power = current_power
                    audio_scaled = audio.copy()

                else:
                    # 根据目标功率和 db_ratio 缩放干扰音频
                    if target_power > 0 and current_power > 0:
                        scalar = (10 ** (db_ratio / 20)) * np.sqrt(target_power / current_power)
                        audio_scaled = audio * scalar
                    else:
                        print(f"Warning: Zero power detected for audio {audio_path} or target. Skipping scaling.")
                        audio_scaled = audio * 0

                all_scaled_audios.append(audio_scaled)

            except FileNotFoundError:
                print(f"Warning: Audio file not found at {audio_path}. Skipping this speaker.")
                all_scaled_audios.append(np.zeros(target_samples, dtype=np.float64))
                continue
            except Exception as e:
                print(f"Error processing audio {audio_path}: {e}. Skipping this speaker.")
                all_scaled_audios.append(np.zeros(target_samples, dtype=np.float64))
                continue

        # --- Sum all scaled audios (确保所有 C 个说话人信息都已处理) ---
        if all_scaled_audios and len(all_scaled_audios) == args.C:
            audio_mix = np.sum(np.stack(all_scaled_audios, axis=0), axis=0)
        else:
            continue

        # --- Final Normalization and Write to File ---
        max_abs_val = np.max(np.abs(audio_mix))
        if max_abs_val > 0:
            audio_mix = np.divide(audio_mix, max_abs_val)

        write_wav(mixture_save_path, audio_mix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'Generate N-speaker audio mixtures based on CSV list')
    parser.add_argument('--C', type=int, required=True, help='Number of speakers in the mixture.')
    parser.add_argument('--audio_data_direc', type=str, required=True,
                        help='Directory containing the individual speaker WAV files.')
    parser.add_argument('--mixture_audio_direc', type=str, required=True,
                        help='Output directory for the generated mixture WAV files.')
    parser.add_argument('--mixture_data_list', type=str, required=True,
                        help='Input CSV file containing the list of mixtures to create.')
    args = parser.parse_args()
    main(args)