import os
import numpy as np
import argparse
import csv
import tqdm
import librosa
import scipy.io.wavfile as wavfile
from multiprocessing import Pool

# Set random seed for reproducibility
np.random.seed(0)


# --- NOTE: Helper functions (extract_wav_from_mp4, read_wav, etc.) remain unchanged ---

def extract_wav_from_mp4(line, args):
    """
    Extracts .wav file from mp4 and returns the sample length.
    Note: The 'line' format is assumed to be [split, speaker_id, utt_path].
    """
    # Construct paths
    video_from_path = os.path.join(args.data_direc, line[0], line[1], line[2] + '.mp4')
    audio_save_path = os.path.join(args.audio_data_direc, line[0], line[1], line[2] + '.wav')

    # Ensure directory exists
    audio_dir = os.path.dirname(audio_save_path)
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    # Extract audio if it doesn't exist
    if not os.path.exists(audio_save_path):
        if os.path.exists(video_from_path):
            os.system(
                f"ffmpeg -i {video_from_path} -y -vn -acodec pcm_s16le -ar {args.sampling_rate} {audio_save_path} > /dev/null 2>&1")
        else:
            return 0

    try:
        sr, audio = wavfile.read(audio_save_path)
        assert sr == args.sampling_rate, "Sampling rate mismatch"
        sample_length = audio.shape[0]
        return sample_length
    except Exception as e:
        return 0


def main(args):
    target_length = float(args.min_length)

    train_list = []
    val_list = []
    test_list = []
    tmp_list = []

    print("Gathering file names and extracting audios...")

    # --- Helper function to process file lists ---
    def process_split(split_name, target_list):
        split_path = os.path.join(args.data_direc, split_name)
        if not os.path.isdir(split_path):
            return

        for root, dirs, files in os.walk(split_path):
            for filename in files:
                if filename.endswith('.mp4'):
                    rel_path_to_data_direc = os.path.relpath(root, args.data_direc)
                    parts = rel_path_to_data_direc.split(os.sep)

                    if len(parts) < 3: continue

                    split = parts[0]
                    speaker_id = parts[1]
                    utt_path = os.path.join(parts[2], filename.split('.')[0])

                    # ln: [split, speaker_id, utt_path]
                    ln = [split, speaker_id, utt_path]

                    sample_length = extract_wav_from_mp4(ln, args)

                    if sample_length < args.min_length * args.sampling_rate: continue

                    ln.append(sample_length / args.sampling_rate)
                    target_list.append(ln)

    # Get test set list of audios
    process_split('test', test_list)
    # Get train/val set list of audios (initially into tmp_list)
    process_split('train', tmp_list)

    print(f"Total audios found: {len(test_list) + len(tmp_list)}. Min length {args.min_length}s applied.")

    # --- Speaker Filtering and Train/Val Split logic remains unchanged ---
    speakers = {}
    for ln in tmp_list:
        ID = ln[1]
        speakers[ID] = speakers.get(ID, 0) + 1
    sort_speakers = sorted(speakers.items(), key=lambda x: x[1], reverse=True)

    train_speakers = {}
    for i, (ID, count) in enumerate(sort_speakers):
        if i == 800: break
        train_speakers[ID] = 0

    for ln in tmp_list:
        ID = ln[1]
        if ID in train_speakers:
            current_count = train_speakers[ID]
            if current_count < 12:
                val_list.append(ln)
                train_speakers[ID] += 1
            elif current_count >= 62:
                continue
            else:
                train_list.append(ln)
                train_speakers[ID] += 1

    # --- Create mixture list ---
    print("Creating mixture list...")
    with open(args.mixture_data_list, 'w', newline='') as f:
        w = csv.writer(f)

        # Iterate over train and validation sets
        for data_list, num_samples in [
            (train_list, args.train_samples),
            (val_list, args.val_samples)
        ]:
            if not data_list: continue

            # Determine the split name of the mixture (e.g., 'train' or 'val')
            # Assuming 'val' is part of 'train' split directory, we use this logic:
            mix_split_name = 'train' if len(data_list) > 10000 else 'val'
            length = num_samples

            unique_speakers = len(set(ln[1] for ln in data_list))
            print(
                f"In {mix_split_name} list: {unique_speakers} speakers, {len(data_list)} utterances. Target mixtures: {length}")

            cache_list = data_list[:]
            count = 0

            while count < length:
                current_pool = cache_list if len(cache_list) >= args.C else data_list

                # --- CRITICAL CHANGE 1: Start mixtures list with the mix_split_name ---
                mixtures = [mix_split_name]
                # --------------------------------------------------------------------

                speaker_ids_in_mix = set()
                speakers_selected = []

                # Select C unique speakers/utterances
                while len(speakers_selected) < args.C:
                    if not current_pool: break

                    idx = np.random.randint(0, len(current_pool))
                    selected_utterance = current_pool[idx]
                    current_speaker_id = selected_utterance[1]

                    if current_speaker_id not in speaker_ids_in_mix:
                        speaker_ids_in_mix.add(current_speaker_id)
                        speakers_selected.append(selected_utterance)

                        if current_pool is cache_list and len(cache_list) >= args.C:
                            cache_list.pop(idx)

                    if len(speakers_selected) == args.C: break

                if len(speakers_selected) < args.C:
                    if count == 0:
                        print(
                            f"Skipping {mix_split_name} mixture generation: only {len(speakers_selected)} speakers found.")
                    break

                    # Populate the rest of the CSV row
                for i, ln in enumerate(speakers_selected):
                    # ln is [split, speaker_id, utt_path, sample_length_seconds]

                    # --- CRITICAL CHANGE 2: Prepend each speaker block with its Split Name (ln[0]) ---
                    mixtures.append(ln[0])  # [0] Split Name (train/val)
                    mixtures.append(ln[1])  # [1] speaker_id
                    mixtures.append(ln[2])  # [2] utt_path

                    # Add db_ratio
                    if i == 0:
                        db_ratio = 0.0
                    else:
                        db_ratio = np.random.uniform(-args.mix_db, args.mix_db)
                    mixtures.append(db_ratio)  # [3] db_ratio

                mixtures.append(target_length)  # Append the unified target length
                w.writerow(mixtures)
                count += 1

        # --- Test Set Generation ---
        data_list = test_list
        length = args.test_samples
        mix_split_name = 'test'

        unique_speakers = len(set(ln[1] for ln in data_list))
        print(
            f"In {mix_split_name} list: {unique_speakers} speakers, {len(data_list)} utterances. Target mixtures: {length}")

        for count in range(length):
            # --- CRITICAL CHANGE 1: Start mixtures list with the mix_split_name ---
            mixtures = [mix_split_name]
            # --------------------------------------------------------------------

            speaker_ids_in_mix = set()
            speakers_selected = []

            # Select C unique speakers/utterances from test_list (sampling with replacement)
            while len(speakers_selected) < args.C:
                idx = np.random.randint(0, len(data_list))
                selected_utterance = data_list[idx]
                current_speaker_id = selected_utterance[1]

                if current_speaker_id not in speaker_ids_in_mix:
                    speaker_ids_in_mix.add(current_speaker_id)
                    speakers_selected.append(selected_utterance)

            # Populate the CSV row
            for i, ln in enumerate(speakers_selected):
                # ln is [split, speaker_id, utt_path, sample_length_seconds]

                # --- CRITICAL CHANGE 2: Prepend each speaker block with its Split Name (ln[0]) ---
                mixtures.append(ln[0])  # [0] Split Name (test)
                mixtures.append(ln[1])  # [1] speaker_id
                mixtures.append(ln[2])  # [2] utt_path

                if i == 0:
                    db_ratio = 0.0
                else:
                    db_ratio = np.random.uniform(-args.mix_db, args.mix_db)
                mixtures.append(db_ratio)  # [3] db_ratio

            mixtures.append(target_length)  # Append the unified target length
            w.writerow(mixtures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LRS2/VoxCeleb dataset mixture list creation')
    parser.add_argument('--data_direc', type=str, required=True)
    parser.add_argument('--C', type=int, required=True, help='Number of speakers in the mixture.')
    parser.add_argument('--mix_db', type=float, required=True, help='Random db ratio range from -mix_db to +mix_db.')
    parser.add_argument('--train_samples', type=int, required=True)
    parser.add_argument('--val_samples', type=int, required=True)
    parser.add_argument('--test_samples', type=int, required=True)
    parser.add_argument('--audio_data_direc', type=str, required=True)
    parser.add_argument('--min_length', type=int, required=True)
    parser.add_argument('--sampling_rate', type=int, required=True)
    parser.add_argument('--mixture_data_list', type=str, required=True)
    args = parser.parse_args()

    main(args)