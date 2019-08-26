from tensorflow_model import TensorflowModel
import tensorflow as tf
from omr_input_adapter import OMRInputAdapter
from omr_output_adapter import OMROutputAdapter
from typing import List

class OMRModelPrediction(TensorflowModel):
    def __init__(
        self, 
        saved_model_filepath: str):
        super().__init__()
        self._INPUT_TENSOR_NAME = 'model_input'
        self._SEQUENCE_LENGTHS_TENSOR_NAME = 'seq_lengths'
        self._KEEP_PROBABILITY_TENSOR_NAME = 'keep_prob'
        self._HEIGHT_TENSOR_NAME = 'input_height'
        self._WIDTH_REDUCTION_TENSOR_NAME = 'width_reduction'
        self._LOGITS_COLLECTION_NAME = 'logits'
        self._OUTPUT_TENSOR_NAME = 'target'
        
        if saved_model_filepath:
            self._load_model_from_file(saved_model_filepath)

    def _load_model_from_file(
        self, 
        saved_model_filepath: str):
        self._sess = super()._restore_saved_model(saved_model_filepath)
        graph = tf.get_default_graph()

        self._input, self._seq_len, self._rnn_keep_prob, height_tensor, width_reduction_tensor = super()._get_tensors_by_name(
            graph, 
            [self._INPUT_TENSOR_NAME, 
            self._SEQUENCE_LENGTHS_TENSOR_NAME, 
            self._KEEP_PROBABILITY_TENSOR_NAME, 
            self._HEIGHT_TENSOR_NAME, 
            self._WIDTH_REDUCTION_TENSOR_NAME])

        logits = graph.get_collection(self._LOGITS_COLLECTION_NAME)[0]

        self._WIDTH_REDUCTION, self._HEIGHT = self._sess.run([width_reduction_tensor, height_tensor])  #maybe don't have to do this?

        self._decoded, _ = tf.nn.ctc_greedy_decoder(
            logits, 
            self._seq_len)

    def predict(
        self, 
        input_path: str, 
        input_adapter: OMRInputAdapter, 
        output_adapter: OMROutputAdapter) -> List[str]:
        image = input_adapter.encode_input(
            input_path, 
            self._HEIGHT)

        seq_lengths = [image.shape[2] / self._WIDTH_REDUCTION]

        prediction = self._sess.run(self._decoded,
                      feed_dict={
                          self._input: image,
                          self._seq_len: seq_lengths,
                          self._rnn_keep_prob: 1.0,  #remove dropout for prediction
                      })

        return output_adapter.decode_prediction(prediction)