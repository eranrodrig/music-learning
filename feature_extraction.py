from kymatio import Scattering1D
from os import listdir
import librosa
import torch

frame_length = 8158
SAMPLE_DATA = 0
SAMPLE_LABEL = 1

genre_dictionary = {
    'classical': 1, 'hiphop': 2, 'jazz': 3, 'metal': 4, 'pop': 5, 'reggae': 6
}


def extract_raw_features(track_file_path):
    raw_track, _ = librosa.load(track_file_path)

    return raw_track


def split_sample(sample):
    sample_track = sample[SAMPLE_DATA]
    splitted_track = [sample_track[i:i + frame_length]
                      for i in range(0, len(sample_track), frame_length)]

    return map(
        lambda track_part: (track_part, sample[SAMPLE_LABEL]), splitted_track)


def to_normalized_tensor(sample_data):
    sample_data_tensor = torch.Tensor(sample_data).float()
    sample_data_tensor /= sample_data_tensor.abs().max()
    sample_data_tensor = sample_data_tensor.view(1, -1)

    return sample_data_tensor


def load_dataset(dataset_path):
    scattering_function = Scattering1D(6, frame_length, 8)
    all_tracks_names = listdir(dataset_path)
    labels = map(
        lambda track_file_name: genre_dictionary[track_file_name.split('.')[0]], all_tracks_names)
    raw_data_samples = map(
        lambda track_file_name, label: (extract_raw_features(
            dataset_path + '/' + track_file_name), label),
        all_tracks_names, labels)
    raw_data_splitted_samples = [item for sublist in map(
        split_sample, raw_data_samples) for item in sublist]
    uni_sized_data_splitted_samples = filter(
        lambda sample:
            len(sample[SAMPLE_DATA]) == frame_length,
        raw_data_splitted_samples)
    tensor_data_samples = map(lambda sample: (
        to_normalized_tensor(sample[SAMPLE_DATA]),
        sample[SAMPLE_LABEL]
    ), uni_sized_data_splitted_samples)
    scattered_data_samples = map(
        lambda sample: (scattering_function.forward(
            sample[SAMPLE_DATA]), sample[SAMPLE_LABEL]),
        tensor_data_samples)

    return list(scattered_data_samples)
