from feature_extraction import load_dataset
import pickle

pickle.dump(load_dataset('./gtzan/train'), open('./gtzan/train-scattered', 'wb'))
pickle.dump(load_dataset('./gtzan/test'), open('./gtzan/test-scattered', 'wb'))
pickle.dump(load_dataset('./gtzan/validation'), open('./gtzan/validation-scattered', 'wb'))
