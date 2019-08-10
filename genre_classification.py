from numpy import array
from feature_extraction import load_scattered_datasets
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from settings import frame_length, SAMPLE_DATA, AUTO_ENCODER_PATH
import pickle


datasets = load_scattered_datasets()
trainning_data_samples = array([sample[SAMPLE_DATA].numpy()
                                for sample in datasets['train']])
trainning_data_samples = trainning_data_samples.reshape(
    (len(trainning_data_samples), 16002, 1))

model = Sequential()

model.add(LSTM(100, activation='relu', input_shape=(16002, 1)))
model.add(RepeatVector(16002))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))

model.compile(optimizer='adam', loss='mse')


model.fit(trainning_data_samples,
          trainning_data_samples, epochs=300, verbose=1)
model.save(AUTO_ENCODER_PATH)
