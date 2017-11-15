#!/usr/bin/env python2.7
"""Read checkpoint and export model.

Usage: lead_saved.py [--model_version=x] [--checkpoint_dir=y] export_dir
"""

import os.path
import sys
import tensorflow as tf 
import numpy as np
from tensorflow.contrib import learn

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('checkpoint_dir', None, 'checkpoint directory.')

# Model Hyperparameters
tf.app.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.app.flags.DEFINE_string("filter_sizes", 5, "Comma-separated filter sizes (default: '3,4,5')")
tf.app.flags.DEFINE_integer("num_filters", 5, "Number of filters per filter size (default: 128)")

FLAGS = tf.app.flags.FLAGS

NUM_TOP_CLASSES = 2

def export():
	if len(sys.argv) < 3 or sys.argv[-1].startswith('-'):
		print('Usage: lead_prediction_saved.py [--model_version=x] [--checkpoint_dir=y] export_dir')
		sys.exit(-1)
	if FLAGS.model_version <= 0:
		print ('Please specify a positive value for version number.')
		sys.exit(-1)

	# Read the lasted checkpoint and vocab of the model
	checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

	vocab_file = os.path.join(FLAGS.checkpoint_dir, "vocab")
	vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_file)
	test = "tmp"
	x_test = np.array(list(vocab_processor.transform(test)))
	sequence_length = x_test.shape[1]

	graph = tf.Graph()
	with graph.as_default():
		# Input transformation.
		serialized_tf_example = tf.placeholder(tf.string,name='tf_example')
		feature_configs = {'input_x': tf.FixedLenFeature(shape=[sequence_length], dtype=tf.float32)}
		tf_example = tf.parse_example(serialized_tf_example, feature_configs)
		input_x = tf.identity(tf_example['input_x'],name='input_x')

		# Run inference.
		with tf.Session() as sess:
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)
			
			# Read embedding layer
			W_embedding = graph.get_operation_by_name("embedding/W").outputs[0]
			embedded_chars = tf.nn.embedding_lookup(W_embedding, tf.to_int32(input_x))
			embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

			# Read convolution + maxpool layer
			filter_shape = [FLAGS.filter_sizes, FLAGS.embedding_dim, 1, FLAGS.num_filters]
			W_conv_maxpool_5 = graph.get_operation_by_name("conv-maxpool-5/W").outputs[0]
			b_conv_maxpool_5 = graph.get_operation_by_name("conv-maxpool-5/b").outputs[0]
			conv_maxpool_5 = tf.nn.conv2d(embedded_chars_expanded, W_conv_maxpool_5, strides=[1,1,1,1], padding="VALID")
			h_conv_maxpool_5 = tf.nn.relu(tf.nn.bias_add(conv_maxpool_5, b_conv_maxpool_5))
			pooled_conv_maxpool_5 = tf.nn.max_pool(h_conv_maxpool_5,ksize=[1,sequence_length-FLAGS.filter_sizes+1, 1, 1], strides=[1,1,1,1], padding="VALID")
			h_pool_flat = tf.reshape(pooled_conv_maxpool_5,[-1,FLAGS.num_filters])

			# Read final score and prediction layer
			W_output = graph.get_operation_by_name("output/W").outputs[0]
			b_output = graph.get_operation_by_name("output/b").outputs[0]
			scores = tf.nn.xw_plus_b(h_pool_flat, W_output, b_output)
			final_scores = tf.nn.softmax(scores)

			values, indices = tf.nn.top_k(final_scores, NUM_TOP_CLASSES)
			table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant(["Bad", "Good"]))
			prediction_classes = table.lookup(tf.to_int64(indices))

			# Export inference model.
			export_path_base = sys.argv[-1]
			output_path = os.path.join(tf.compat.as_bytes(export_path_base),tf.compat.as_bytes(str(FLAGS.model_version)))
			print("Exporting trained model to {}".format(output_path))
			builder = tf.saved_model.builder.SavedModelBuilder(output_path)

			# Build the signature_def_map.
			classify_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
			classes_output_tensor_info = tf.saved_model.utils.build_tensor_info(prediction_classes)
			scores_output_tensor_info = tf.saved_model.utils.build_tensor_info(values)

			classification_signature = (tf.saved_model.signature_def_utils.build_signature_def(inputs={tf.saved_model.signature_constants.CLASSIFY_INPUTS:classify_inputs_tensor_info},
																								outputs={tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:classes_output_tensor_info,
																										tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:scores_output_tensor_info},
																								method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))
			predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(input_x)
			prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(inputs={'texts':predict_inputs_tensor_info},
																							outputs={'classes': classes_output_tensor_info,
																									'scores':scores_output_tensor_info},
																							method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
			legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
			builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],signature_def_map={'predict_texts':prediction_signature, 
																							tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:classification_signature},
																							legacy_init_op=legacy_init_op)
			builder.save()
			print ("Done Exporting!")


def main(unused_argv=None):
	export()


if __name__ == '__main__':
	tf.app.run()

	