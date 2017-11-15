# Nolo Lead Quality Prediction in TensorFlow and TensorFlow serving (Divorce AoP only)


## Getting Started

### Install prerequisites in Centos 7

Follow instructions in install.sh

### Clone the TensorFlow serving repositiory
```
$>git clone --recurse-submodules https://github.com/tensorflow/serving
$>cd serving
```
### Configure TensorFlow
```
$>cd tensorflow
$>./configure
$>cd ..
```
### Clone this repo in /serving/tensorflow_serving directory
```
$>cd tensorflow_serving
$>git clone git@git.internetbrands.com:data-science/Nolo_Lead_Quality_Prediction.git
```
### Download [GoogleNews-vectors-negative300](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) and save it in /serving/tensorflow_serving/lead_model_serving directory



## Start training the model and save the checkpoints locally
```
$>bazel build -c opt //tensorflow_serving/lead_model_serving:lead_training
$>bazel-bin/tensorflow_serving/lead_model_serving/lead_training --positive_data_file=/serving/tensorflow_serving/lead_model_serving/data/data_original/good_leads.txt --negative_data_file=/serving/tensorflow_serving/lead_model_serving/data/data_original/bad_leads.txt  --model_base_dir=/serving/tensorflow_serving/lead_model_serving --checkpoint_version=1 --word2vec=/home/mcong/serving/tensorflow_serving/lead_model_serving/GoogleNews-vectors-negative300.bin
```
### Needed parameters:
* positive_data_file: Path to the data with good leads
* negative_data_file: Path to the data with bad leads
* model_base_dir: Path to the base of the model
* checkpoint_version: Checkpoint version
* word2vec: Path to the pretrained word2vec



## Load the checkpoint, add the signiture and export the model for serving
```
$>bazel build -c opt //tensorflow_serving/lead_model_serving:lead_saved
$>bazel-bin/tensorflow_serving/lead_model_serving/lead_saved --model_version=1 --checkpoint_dir=/serving/tensorflow_serving/lead_model_serving/checkpoints/1 /serving/tensorflow_serving/lead_model_serving/exported_models
```
### Needed parameters:
* model_version: Model version to be saved
* checkpoint_dir: Path to the checkpoint
* export_dir: Path to texport the model (don't need specify the flag name)



## Load Exported Model with Standard TensorFlow ModelServer
```
$>sudo touch /usr/include/stropts.h
$>bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server
$>bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=lead_quality_prediction --model_base_path=/home/mcong/serving/tensorflow_serving/lead_model_serving/exported_models
```


## Test the Serving
```
$>bazel build -c opt //tensorflow_serving/lead_model_serving:lead_client
$>bazel-bin/tensorflow_serving/lead_model_serving/lead_client --server=localhost:9000 --text_file=/serving/tensorflow_serving/lead_model_serving/example.txt --vocab_dir=/serving/tensorflow_serving/lead_model_serving/checkpoints/1/vocab
```


## References

* [TensorFlow Serving](https://www.tensorflow.org/serving/)
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
* [Implementing a CNN for Text Classification in Tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
* [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
* [Load Pretrained word2vec into cnn-text-classification-tf](https://gist.github.com/j314erre/b7c97580a660ead82022625ff7a644d8)
