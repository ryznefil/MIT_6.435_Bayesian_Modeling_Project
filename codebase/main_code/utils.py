import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def pickle_object(file, save_location):
    with open(save_location, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_object(file_location: object) -> object:
    with open(file_location, 'rb') as handle:
        file = pickle.load(handle)
    return file
