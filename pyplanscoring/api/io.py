import bz2
import pickle


# DVH OBJECTS
def save(obj, filename, protocol=-1):
    """
        Saves  Object into a file using gzip and Pickle
    :param obj: Calibration Object
    :param filename: Filename *.fco
    :param protocol: cPickle protocol
    """
    with bz2.BZ2File(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)


def load(filename):
    """
        Loads a Calibration Object into a file using gzip and Pickle
    :param filename: Calibration filemane *.fco
    :return: object
    """
    with bz2.BZ2File(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

    # DICOM OBJECTS
    # Report objects PDF files.
