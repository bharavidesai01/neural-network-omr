from typing import List

class OMROutputAdapter:
    def __init__(
        self,
        vocabulary_file_path: str):
        with open(vocabulary_file_path, 'r') as vocabulary_file:
            vocabulary = vocabulary_file.read().splitlines()
            self._words = dict()
            for word in vocabulary:
                index = len(self._words)
                self._words[index] = word

    def decode_prediction(self, prediction) -> List:
        predictions = self.sparse_tensor_to_strings(prediction)
        return [self._words[w] for w in predictions[0]]

    def sparse_tensor_to_strings(
        self, 
        sparse_tensor):
        indices = sparse_tensor[0][0]
        values = sparse_tensor[0][1]
        dense_shape = sparse_tensor[0][2]

        strings = [[] for i in range(dense_shape[0])]
        string = []
        last_index = 0

        for i in range(len(indices)):
            if indices[i][0] != last_index:
                strings[last_index] = string
                string = []
                last_index = indices[i][0]
            string.append(values[i])
        strings[last_index] = string
        return strings