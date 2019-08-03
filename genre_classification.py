from feature_extraction import load_dataset
import pickle

pickle.dump(load_dataset('./gtzan/train'), open('train', 'wb'))
pickle.dump(load_dataset('./gtzan/test'), open('test', 'wb'))
pickle.dump(load_dataset('./gtzan/validation'), open('validation', 'wb'))
