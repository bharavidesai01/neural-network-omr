import tensorflow as tf
import tensorflow_model
from tensorflow import Tensor
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from omr_model_hyperparameters import OMRModelHyperparameters
from omr_output_adapter import OMROutputAdapter
from omr_input_adapter import OMRInputAdapter
from omr_cross_validation import OMRCrossValidation
from tensorflow_model import TensorflowModel
from typing import List
import os
import numpy as np


class OMRModelTraining(TensorflowModel):
    def __init__(self):
        self._INPUT_TENSOR_NAME = 'model_input'
        self._SEQUENCE_LENGTHS_TENSOR_NAME = 'seq_lengths'
        self._KEEP_PROBABILITY_TENSOR_NAME = 'keep_prob'
        self._HEIGHT_TENSOR_NAME = 'input_height'
        self._WIDTH_REDUCTION_TENSOR_NAME = 'width_reduction'
        self._LOGITS_COLLECTION_NAME = 'logits'
        self._OUTPUT_TENSOR_NAME = 'target'
        
    def train(
        self,
        hyperparameters: OMRModelHyperparameters,
        primus,
        output_adapter: OMROutputAdapter,
        input_adapter: OMRInputAdapter,
        validator: OMRCrossValidation,
        save_location: str,
        log_file_location='./graphs') -> None:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        sess = tf.InteractiveSession(config=config)

        inputs, seq_len, labels, output, loss, rnn_keep_probability = self._build_graph(hyperparameters)
        training_optimizer = tf.train.AdamOptimizer().minimize(loss)

        saver = tf.train.Saver(max_to_keep=None)

        tf.summary.scalar(name='loss', tensor=loss)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
            log_file_location, 
            sess.graph)

        sess.run(tf.global_variables_initializer())

        # Training loop
        for epoch in range(hyperparameters.max_epochs):
            batch = primus.nextBatch(hyperparameters)

            _, loss_value, summary = sess.run([training_optimizer, loss, merged],
                                    feed_dict={
                                        inputs: batch['inputs'],
                                        seq_len: batch['seq_lengths'],
                                        labels: input_adapter.sparse_tuple_from_sequence(batch['targets']),
                                        rnn_keep_probability: hyperparameters.keep_probability,
                                    })

            print ('Loss value at epoch {0}/{1}: {2} '.format(
                str(epoch),
                hyperparameters.max_epochs,
                str(loss_value)))
            #print(batch['image_names'])

            writer.add_summary(summary, epoch)

            if epoch % 1000 == 0:
                print ('Validating...')

                validation_batch, validation_size = primus.getValidation(hyperparameters)
                
                val_idx = 0
                
                val_ed = 0
                val_len = 0
                val_count = 0
                    
                while val_idx < validation_size:
                    mini_batch_feed_dict = {
                        inputs: validation_batch['inputs'][val_idx:val_idx+hyperparameters.minibatch_size],
                        seq_len: validation_batch['seq_lengths'][val_idx:val_idx+hyperparameters.minibatch_size],
                        rnn_keep_probability: 1.0            
                    }            
                                
                    print('Validating minibatch {0} of {1}'.format(val_idx, validation_size))
                    prediction = sess.run(
                        output,
                        mini_batch_feed_dict)
            
                    str_predictions = output_adapter.sparse_tensor_to_strings(prediction)
            

                    for i in range(len(str_predictions)):
                        ed = validator.edit_distance(str_predictions[i], validation_batch['targets'][val_idx+i])
                        val_ed = val_ed + ed
                        val_len = val_len + len(validation_batch['targets'][val_idx+i])
                        val_count = val_count + 1
                        
                    val_idx = val_idx + hyperparameters.minibatch_size
            
                print ('[Epoch ' + str(epoch) + '] ' + str(1. * val_ed / val_count) + ' (' + str(100. * val_ed / val_len) + ' SER) from ' + str(val_count) + ' samples.')        
                print ('Saving the model...')
                saver.save(sess,save_location,global_step=epoch)
                print ('------------------------------')

    def _build_graph(
        self, 
        params: OMRModelHyperparameters):
        input = tf.placeholder(shape=(None,
                                    params.image_height,
                                    params.image_width,
                                    params.image_channels),  # [batch, height, width, channels]
                                dtype=tf.float32,
                                name=self._INPUT_TENSOR_NAME)

        width_reduction = 1  # todo this can be precalculated in the hyperparameters. no need to store in model
        height_reduction = 1

        # Convolutional blocks
        x = input
        for i in range(len(params.convolutional_filter_number)):

            x = tf.layers.conv2d(
                inputs=x,
                filters=params.convolutional_filter_number[i],
                kernel_size=params.convolutional_filter_size[i],
                padding='same',
                activation=None)

            x = tf.layers.batch_normalization(x)
            x = super()._leaky_relu(x)

            x = tf.layers.max_pooling2d(
                inputs=x,
                pool_size=params.convolutional_pooling_size[i],
                strides=params.convolutional_pooling_size[i])

            width_reduction = width_reduction * params.convolutional_pooling_size[i][1]
            height_reduction = height_reduction * params.convolutional_pooling_size[i][0]


        # Prepare output of conv block for recurrent blocks
        features = tf.transpose(x, perm=[2, 0, 3, 1])  # -> [width, batch, height, channels] (time_major=True)
        feature_dim = params.convolutional_filter_number[-1] * (params.image_height / height_reduction)
        feature_width = tf.shape(input)[2] / width_reduction
        features = tf.reshape(features, tf.stack([tf.cast(feature_width, 'int32'), tf.shape(input)[0], tf.cast(feature_dim,'int32')]))  # -> [width, batch, features]

        # Store metadata in the model about scaling factors
        tf.constant(params.image_height, name=self._HEIGHT_TENSOR_NAME)
        tf.constant(width_reduction, name=self._WIDTH_REDUCTION_TENSOR_NAME)

        # Recurrent block
        rnn_keep_prob = tf.placeholder(dtype=tf.float32, name=self._KEEP_PROBABILITY_TENSOR_NAME)
        rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            tf.contrib.rnn.MultiRNNCell(
                [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(params.rnn_units_per_cell), input_keep_prob=rnn_keep_prob)
                for _ in range(params.rnn_cells)]),
            tf.contrib.rnn.MultiRNNCell(
                [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(params.rnn_units_per_cell), input_keep_prob=rnn_keep_prob)
                for _ in range(params.rnn_cells)]),
            features,
            dtype=tf.float32,
            time_major=True,
        )

        rnn_outputs = tf.concat(rnn_outputs, 2)

        logits = tf.contrib.layers.fully_connected(
            rnn_outputs,
            params.vocabulary_size + 1,  # BLANK
            activation_fn=None,
        )
        
        tf.add_to_collection(self._LOGITS_COLLECTION_NAME, logits) # for restoring purposes

        # CTC Loss computation
        seq_len = tf.placeholder(tf.int32, [None], name=self._SEQUENCE_LENGTHS_TENSOR_NAME)
        labels = tf.sparse_placeholder(dtype=tf.int32, name=self._OUTPUT_TENSOR_NAME)
        ctc_loss = tf.nn.ctc_loss(labels=labels, inputs=logits, sequence_length=seq_len, time_major=True)
        loss = tf.reduce_mean(ctc_loss)

        # CTC decoding
        decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

        return input, seq_len, labels, decoded, loss, rnn_keep_prob