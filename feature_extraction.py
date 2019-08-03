from kymatio import Scattering1D
from os import listdir
from numpy import fromfile
import librosa
import torch

frame_length = 370
sample_data = 0
sample_label = 1

genre_dictionary = {
    'classical': 1, 'hiphop': 2, 'jazz': 3, 'metal': 4, 'pop': 5, 'reggae': 6
}


def extract_raw_features(track_file_path):
    raw_track, _ = librosa.load(track_file_path)

    return raw_track


def split_sample(sample):
    splitted_track = librosa.effects.split(
        sample[sample_data], frame_length=frame_length)

    return map(lambda track_frame: (track_frame, sample[sample_label]), splitted_track)


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
    tensor_data_samples = map(lambda sample: (torch.tensor(
        sample[sample_data]).view(1, -1), sample[sample_label]), raw_data_splitted_samples)
    scattered_data_samples = map(
        lambda sample: (scattering_function.forward(
            sample[sample_data]), sample[sample_label]),
        tensor_data_samples)
        
    return list(scattered_data_samples)
