import cv2
import numpy as np

class OMRInputAdapter:
    def __init__(self):
        pass

    def encode_input(
        self, 
        image_file_path: str, 
        image_height: float):
        image = cv2.imread(image_file_path, False)
        image = self.resize(image, image_height)
        image = self.normalize(image)
        image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)
        return image

    def normalize(
        self, 
        image):
        return (255. - image)/255.

    def resize(
        self, 
        image, 
        height):
        width = int(float(height * image.shape[1]) / image.shape[0])
        return cv2.resize(image, (width, height))

    def sparse_tuple_from_sequences(
        self,
        sequences, 
        dtype=np.int32):
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape