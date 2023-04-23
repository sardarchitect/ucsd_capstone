import cv2 as cv

def dict_key_string_to_int(dictionary):
    # Util function that converts all dictionary keys from str to int
    dictionary= {int(k):v for k, v in dictionary.items()}
    return dictionary

def rescaleFrame(frame, scale=0.25):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
