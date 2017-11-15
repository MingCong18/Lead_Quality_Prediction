# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7


from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf
import re
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib import learn
import numpy as np

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('text_file', '/serving/tensorflow_serving/lead_serving1/try.txt', 'path to the text file')
tf.app.flags.DEFINE_string('vocab_dir', '/serving/tensorflow_serving/lead_serving1/1500351148/vocab', 'path to vocab dir')
FLAGS = tf.app.flags.FLAGS

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  # Send request
  #with open(FLAGS.text_file, 'r') as f:
    # See prediction_service.proto for gRPC request/response details.
    #data = f.read()
  data = list(open(FLAGS.text_file,'r').readlines())
  data = [s.strip() for s in data]
  data = [clean_str(i) for i in data]
  vocab_processor = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_dir)
  data = np.array(list(vocab_processor.transform(data)))
  data = np.array(data,dtype=np.float32)
  #data = np.array(data,dtype=np.int32)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'lead_quality_prediction'
  request.model_spec.signature_name = 'predict_texts'
  request.inputs['texts'].CopyFrom(tf.contrib.util.make_tensor_proto(data))
  result = stub.Predict(request, 5.0)  # 10 secs timeout
  print(result)
  result_future = stub.Predict.future(request, 5.0)
  prob = np.array(result_future.result().outputs['scores'].float_val)
  response = np.array(result_future.result().outputs['classes'].string_val)
  print("\nCut-off point is 0.8")
  if prob[0]>=0.8:
    final_response = response[0]
  else:
    final_response = "Good"
  print("Final result is {}".format(final_response))


if __name__ == '__main__':
  tf.app.run()
