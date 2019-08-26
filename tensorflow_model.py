import tensorflow as tf
from tensorflow import Tensor
from tensorflow import Graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from typing import List

class TensorflowModel:
    def __init__(self):
        self._GET_TENSOR_FORMAT_STRING = '{0}:0'

    # Restoring Model
    
    def _restore_saved_model(
        self, 
        saved_model_filepath: str):
        tf.reset_default_graph()
        session = tf.InteractiveSession()
        tf.train.import_meta_graph(saved_model_filepath).restore(
            session, 
            saved_model_filepath.split('.')[0])
        return session

    def _get_tensors_by_name(
        self, 
        graph: Graph, 
        names: List[str]) -> List[Tensor]:
        return [self._get_tensor_by_name(graph, name) for name in names]

    def _get_tensor_by_name(
        self, 
        graph: Graph, 
        name: str) -> Tensor:
        return graph.get_tensor_by_name(self._GET_TENSOR_FORMAT_STRING.format(name))

    # End Restoring Model

    # Building Model for Training

    def _leaky_relu(
        self, 
        features, 
        alpha=0.2, 
        name=None) -> Tensor:
        with ops.name_scope(name, 'LeakyRelu', [features, alpha]):
            features = ops.convert_to_tensor(
                features,
                name='features')
            alpha = ops.convert_to_tensor(
                alpha, 
                name='alpha')
            return math_ops.maximum(
                alpha * features, 
                features)
    
    def _add_ctc_loss_with_decoder_to_graph_with_decoder(
        self, 
        inputs: Tensor, 
        sequence_length_tensor_name: str, 
        output_tensor_name: str) -> List[Tensor]:
        sequence_length = tf.placeholder(
            dtype=tf.int32, 
            shape=[None], 
            name=sequence_length_tensor_name)
        labels = tf.sparse_placeholder(
            dtype=tf.int32, 
            name=output_tensor_name)
        loss = tf.reduce_mean(tf.nn.ctc_loss(
            labels=labels, 
            inputs=inputs, 
            sequence_length=sequence_length, 
            time_major=True))
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs, 
            sequence_length)

        return sequence_length, labels, loss, decoded

    def _add_convolutional_layers_to_graph(
        self, 
        input: Tensor, 
        number_of_filters: list, 
        filter_size: list, 
        pooling_size: list) -> Tensor:
        output = input
        for i in range(len(number_of_filters)):
            output = tf.layers.conv2d(
                inputs=output,
                filters=number_of_filters[i],
                kernel_size=filter_size[i],
                padding='same',
                activation=None)
            output = tf.layers.max_pooling2d(
                inputs=self._leaky_relu(tf.layers.batch_normalization(output)),
                pool_size=pooling_size[i],
                strides=pooling_size[i])

        return output

    def _add_recurrent_layers_to_graph(
        self, 
        input: Tensor, 
        keep_probability_tensor_name: str, 
        number_of_recurrent_layers: int, 
        number_of_lstm_units: int) -> List[Tensor]:
        keep_probability = tf.placeholder(
            dtype=tf.float32, 
            name=keep_probability_tensor_name)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            self._create_stacked_dropout_lstm_cells(
                number_of_lstm_units, 
                keep_probability, 
                number_of_recurrent_layers),
            self._create_stacked_dropout_lstm_cells(
                number_of_lstm_units, 
                keep_probability, 
                number_of_recurrent_layers),
            input,
            dtype=tf.float32,
            time_major=True)

        return tf.concat(outputs, 2), keep_probability

    def _create_stacked_dropout_lstm_cells(
        self,
        number_of_lstm_units: int, 
        keep_probability: Tensor, 
        number_of_lstm_cells: int) -> Tensor:
        return tf.contrib.rnn.MultiRNNCell(
                [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(number_of_lstm_units), input_keep_prob=keep_probability)
                for _ in range(number_of_lstm_cells)])

    # End Build Model for Training