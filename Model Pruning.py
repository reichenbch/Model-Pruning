# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import tensorflow as tf
tf.set_random_seed(123)
import numpy as np
np.random.seed(123)

from tensorflow.examples.tutorials.mnist import input_data

# %%
mnist = input_data.read_data_sets("MNIST_data/")

train_data_provider = mnist.train
validation_data_provider = mnist.validation
test_data_provider = mnist.test

# %%
import os
class BaseNetwork:

    @property
    def sess(self):
        if not hasattr(self, '_sess'):
            config = tf.ConfigProto()
            # to save GPU resources
            config.gpu_options.allow_growth = True
            self._sess = tf.Session(config=config, graph=self.graph)
        return self._sess

    def init_variables(self, var_list):

        self.sess.run(tf.variables_initializer(var_list))

    def number_of_parameters(self, var_list):
        return sum(np.prod(v.get_shape().as_list()) for v in var_list)

    def save_model(self, path=None, sess=None, global_step=None, verbose=True):
        save_dir = path or self.model_path
        os.makedirs(save_dir, exist_ok=True)
        self.saver.save(sess or self.sess,
                        os.path.join(save_dir, 'model.ckpt'),
                        global_step=global_step)
        return self

    def load_model(self, path=None, sess=None, verbose=True):
        path = path or self.model_path
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt is None:
            raise FileNotFoundError('Can`t load a model. '\
            'Checkpoint does not exist.')    
        restore_path = ckpt.model_checkpoint_path
        self.saver.restore(sess or self.sess, restore_path)

        return self

# %%
def get_second_dimension(tensor):
    return tensor.get_shape().as_list()[1]

# %%
from typing import Union
from tqdm import tqdm

class FullyConnectedClassifier(BaseNetwork):

    def __init__(self,
                 input_size: int,
                 n_classes: int,
                 layer_sizes: list,
                 model_path: str,
                 activation_fn=tf.nn.relu,
                 dropout=0.25,
                 momentum=0.9,
                 weight_decay=0.0005,
                 scope='FullyConnectedClassifier',
                 verbose=True,
                 pruning_threshold=None):

        """Create an instance of FullyConnectedClassifier"""

        self.input_size = input_size
        self.n_classes = n_classes
        self.layer_sizes = layer_sizes + [n_classes]
        self.model_path = model_path
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scope = scope
        self.verbose = verbose
        self.pruning_threshold = pruning_threshold

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(self.scope):

                self._create_placeholders()

                self.logits = self._build_network(inputs=self.inputs,
                                                  layer_sizes=self.layer_sizes,
                                                  activation_fn=self.activation_fn,
                                                  keep_prob=self.keep_prob)

                self.loss = self._create_loss(logits=self.logits,
                                              labels=self.labels,
                                              weight_decay=self.weight_decay)

                self.train_op = self._create_optimizer(self.loss,
                                                       learning_rate=self.learning_rate,
                                                       momentum=momentum,
                                                       threshold=pruning_threshold)

                self._create_metrics(logits=self.logits,
                                     labels=self.labels,
                                     loss=self.loss)

                self.saver = self._create_saver(tf.global_variables())
                self.init_variables(tf.global_variables())

                if self.verbose:
                    print('\nSuccessfully created graph for {model}.'.format(
                                                                model=self.scope))
                    print('Number of parameters (four bytes == 1 parameter): {}.\n'.format(
                        int(self.number_of_parameters(tf.trainable_variables()))))


    def _create_placeholders(self):

        # create input nodes of a graph
    
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=(None, self.input_size),
                                     name='inputs')
    
        self.labels = tf.placeholder(dtype=tf.int64,
                                     shape=None,
                                     name='labels')
    
        self.keep_prob = tf.placeholder(dtype=tf.float32,
                                        shape=(),
                                        name='keep_prob')
    
        self.learning_rate = tf.placeholder(dtype=tf.float32,
                                            shape=(),
                                            name='learning_rate')
    
    def _build_network(self,
                       inputs: tf.Tensor,
                       layer_sizes: list,
                       activation_fn: callable,
                       keep_prob: Union[tf.Tensor, float]) -> tf.Tensor:

        with tf.variable_scope('network'):
    
            net = inputs
    
            self.weight_matrices = []
            self.biases = []

            weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
            bias_initializer = tf.constant_initializer(0.1)

            # dynamically create a network

            for i, layer_size in enumerate(layer_sizes):
    
                with tf.variable_scope('layer_{layer}'.format(layer=i+1)):

                    name = 'weights'
                    shape = (get_second_dimension(net), layer_size)
                    weights = tf.get_variable(name=name,
                                              shape=shape,
                                              initializer=weights_initializer)

                    self.weight_matrices.append(weights)
                    # L2 loss
                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                         tf.reduce_sum(weights ** 2))
        
                    name = 'bias'
                    shape = [layer_size]
                    bias = tf.get_variable(name=name,
                                           shape=shape,
                                           initializer=bias_initializer)
                    self.biases.append(bias)
    
                    net = tf.matmul(net, weights) + bias
    
                    # no activation and dropout on the last layer
                    if i < len(layer_sizes) - 1:
                        net = activation_fn(net)
                        net = tf.nn.dropout(net, keep_prob=keep_prob)
    
            return net
    
    def _create_loss(self,
                     logits: tf.Tensor,
                     labels: tf.Tensor,
                     weight_decay: float) -> tf.Tensor:
    
        with tf.variable_scope('loss'):
            classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            logits=logits, labels=labels,
                                            name='classification_loss')
    
            classification_loss = tf.reduce_mean(classification_loss,
                                                 name='classification_loss_averaged')

            l2_loss = weight_decay * tf.add_n(tf.losses.get_regularization_losses())
    
            return l2_loss + classification_loss

    def _create_optimizer(self,
                          loss: tf.Tensor,
                          learning_rate: Union[tf.Tensor, float],
                          momentum: Union[tf.Tensor, float],
                          threshold: float) -> tf.Operation:

        if threshold is not None:
            return self._create_optimizer_sparse(loss=loss,
                                                 threshold=threshold,
                                                 learning_rate=learning_rate,
                                                 momentum=momentum)
        with tf.variable_scope('optimizer'):

            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                   momentum=momentum,
                                                   name='optimizer')
            self.global_step = tf.Variable(0)
            train_op = optimizer.minimize(loss,
                                          global_step=self.global_step,
                                          name='train_op')

            return train_op

    def _apply_prune_on_grads(self,
                              grads_and_vars: list,
                              threshold: float):

        # we need to make gradients correspondent
        # to the pruned weights to be zero

        grads_and_vars_sparse = []

        for grad, var in grads_and_vars:
            if 'weights' in var.name:
                small_weights = tf.greater(threshold, tf.abs(var))
                mask = tf.cast(tf.logical_not(small_weights), tf.float32)
                grad = grad * mask

            grads_and_vars_sparse.append((grad, var))
               
        return grads_and_vars_sparse

    def _create_optimizer_sparse(self,
                                 loss: tf.Tensor,
                                 threshold: float,
                                 learning_rate: Union[tf.Tensor, float],
                                 momentum: Union[tf.Tensor, float]) -> tf.Operation:

        with tf.variable_scope('optimizer'):

            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                   momentum=momentum,
                                                   name='optimizer')
            self.global_step = tf.Variable(0)
            grads_and_vars = optimizer.compute_gradients(loss)
            grads_and_vars_sparse = self._apply_prune_on_grads(grads_and_vars,
                                                               threshold)
            train_op = optimizer.apply_gradients(grads_and_vars_sparse,
                                                 global_step=self.global_step,
                                                 name='train_op')

            return train_op

    def _create_metrics(self,
                        logits: tf.Tensor,
                        labels: tf.Tensor,
                        loss: tf.Tensor):

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _create_saver(self, var_list):

        saver = tf.train.Saver(var_list=var_list)
        return saver

    def fit(self,
            n_epochs: int,
            batch_size: int,
            learning_rate_schedule: callable,
            train_data_provider,
            validation_data_provider,
            test_data_provider):

        n_iterations = train_data_provider.num_examples // batch_size

        for epoch in range(n_epochs):
            print('Starting epoch {epoch}.\n'.format(epoch=epoch+1))
            for iteration in tqdm(range(n_iterations), ncols=75):

                images, labels = train_data_provider.next_batch(batch_size)

                feed_dict = {self.inputs: images,
                             self.labels: labels,
                             self.learning_rate: learning_rate_schedule(epoch+1),
                             self.keep_prob: 1 - self.dropout} 

                self.sess.run(self.train_op, feed_dict=feed_dict)
    
            # evaluate metrics after every epoch
            train_accuracy, train_loss = self.evaluate(train_data_provider,
                                                       batch_size=batch_size)
            validation_accuracy, validation_loss = self.evaluate(validation_data_provider,
                                                                 batch_size=batch_size)

            print('\nEpoch {epoch} completed.'.format(epoch=epoch+1))
            print('Accuracy on train: {accuracy}, loss on train: {loss}'.format(
                                    accuracy=train_accuracy, loss=train_loss))
            print('Accuracy on validation: {accuracy}, loss on validation: {loss}'.format(
                                    accuracy=validation_accuracy, loss=validation_loss))

        test_accuracy, test_loss = self.evaluate(test_data_provider,
                                                 batch_size=batch_size)

        print('\nOptimization finished.'.format(epoch=epoch+1))
        print('Accuracy on test: {accuracy}, loss on test: {loss}'.format(
                                accuracy=test_accuracy, loss=test_loss))

        self.save_model(global_step=self.global_step)

    def evaluate(self, data_provider, batch_size: int):

        fetches = [self.accuracy, self.loss]

        n_iterations = data_provider.num_examples // batch_size

        average_accuracy = 0
        average_loss = 0

        for iteration in range(n_iterations):

            images, labels = data_provider.next_batch(batch_size)

            feed_dict = {self.inputs: images,
                         self.labels: labels,
                         self.keep_prob: 1.0} 

            accuracy, loss = self.sess.run(fetches, feed_dict=feed_dict)
            
            average_accuracy += accuracy / n_iterations
            average_loss += loss / n_iterations

        return average_accuracy, average_loss

# %%
class ConfigNetworkDense:

    input_size = 28 * 28
    n_classes = 10
    layer_sizes = [1000, 1000, 500, 200]
    dropout = 0.5
    weight_decay = 0.0005
    activation_fn = tf.nn.relu
    model_path = 'saved_models/network_dense'

    n_epochs = 25
    batch_size = 100

    @staticmethod
    def learning_rate_schedule(epoch):

        if epoch < 10:
            return 1e-2
        elif epoch < 20:
            return 1e-3
        else:
            return 1e-4

class ConfigNetworkDensePruned:

    input_size = 28 * 28
    n_classes = 10
    layer_sizes = [1000, 1000, 500, 200]
    dropout = 0
    weight_decay = 0.0001
    activation_fn = tf.nn.relu
    model_path = 'saved_models/network_dense_pruned'
    pruning_threshold = 0.03

    n_epochs = 20
    batch_size = 100

    @staticmethod
    def learning_rate_schedule(epoch):

        if epoch < 10:
            return 1e-3
        else:
            return 1e-4

class ConfigNetworkSparse:

    input_size = 28 * 28
    n_classes = 10
    activation_fn = tf.nn.relu
    model_path = 'saved_models/network_sparse'
    batch_size = 100

# %%
classifier = FullyConnectedClassifier(input_size=ConfigNetworkDense.input_size,
                                                    n_classes=ConfigNetworkDense.n_classes,
                                                    layer_sizes=ConfigNetworkDense.layer_sizes,
                                                    model_path=ConfigNetworkDense.model_path,
                                                    dropout=ConfigNetworkDense.dropout,
                                                    weight_decay=ConfigNetworkDense.weight_decay,
                                                    activation_fn=ConfigNetworkDense.activation_fn)

# than train it
classifier.fit(n_epochs=ConfigNetworkDense.n_epochs,
               batch_size=ConfigNetworkDense.batch_size,
               learning_rate_schedule=ConfigNetworkDense.learning_rate_schedule,
               train_data_provider=train_data_provider,
               validation_data_provider=validation_data_provider,
test_data_provider=test_data_provider)

# %%
from matplotlib import pyplot as plt

def plot_histogram(weights_list: list,
                   image_name: str,
                   include_zeros=True):

    """A function to plot weights distribution"""

    weights = []
    for w in weights_list:
        weights.extend(list(w.ravel()))

    if not include_zeros:
        weights = [w for w in weights if w != 0]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    ax.hist(weights,
            bins=100,
            facecolor='green',
            edgecolor='black',
            alpha=0.7,
            range=(-0.15, 0.15))

    ax.set_title('Weights distribution')
    ax.set_xlabel('Weights values')
    ax.set_ylabel('Number of weights')

    fig.savefig(image_name + '.png')

# %%
import collections

# Weight Pruning, setting values of the weight of neurons equal to zero
def prune_weights(weights, pruning_threshold):
    small_weights = np.abs(weights) < pruning_threshold
    weights[small_weights] = 0
    values = weights[weights != 0]
    indices = np.transpose(np.nonzero(weights))
    return values, indices

def get_sparse_values_indices(weights):
    values = weights[weights != 0]
    indices = np.transpose(np.nonzero(weights))
    return values, indices

def mask_for_big_values(weights, pruning_threshold):
    small_weights = np.abs(weights) < pruning_threshold
    return np.logical_not(small_weights)

def calculate_number_of_sparse_parameters(sparse_layers):

    total_count = 0

    for layer in sparse_layers:

        total_count += layer.values.nbytes // 4
        total_count += layer.indices.nbytes // 4
        total_count += layer.dense_shape.nbytes // 4
        total_count += layer.bias.nbytes // 4

    return total_count

class SparseLayer(collections.namedtuple('SparseLayer',
                                         ['values',
                                          'indices',
                                          'dense_shape',
                                          'bias'])):

    """An auxilary class to represent sparse layer"""
    pass

# %%
mnist

# %%
train_data_provider

# %% [markdown]
# # Weight Pruning

# %%
def weightPruning(pruning_percent):
    classifier = FullyConnectedClassifier(
                                input_size=ConfigNetworkDensePruned.input_size,
                                n_classes=ConfigNetworkDensePruned.n_classes,
                                layer_sizes=ConfigNetworkDensePruned.layer_sizes,
                                model_path=ConfigNetworkDensePruned.model_path,
                                dropout=ConfigNetworkDensePruned.dropout,
                                weight_decay=ConfigNetworkDensePruned.weight_decay,
                                activation_fn=ConfigNetworkDensePruned.activation_fn,
                                pruning_threshold=ConfigNetworkDensePruned.pruning_threshold)

    # collect tf variables and correspoding optimizer variables
    with classifier.graph.as_default():
        weight_matrices_tf = classifier.weight_matrices
        optimizer_matrices_tf = [v 
                                  for v in tf.global_variables()
                                  for w in weight_matrices_tf
                                  if w.name[:-2] in v.name
                                  and 'optimizer' in v.name]

    # load previously trained model
    # and get values of weights and optimizer variables
    weights, optimizer_weights = (classifier
                                 .load_model(ConfigNetworkDense.model_path)
                                 .sess.run([weight_matrices_tf,
                                            optimizer_matrices_tf]))

    # plot weights distribution before pruning
    weights = classifier.sess.run(weight_matrices_tf)
    plot_histogram(weights,'weights_distribution_before_pruning',include_zeros=False)

    # for each pair (weight matrix + optimizer matrix)
    # get a binary mask to get rid of small values. 
    # Than, based on this mask change the values of 
    # the weight matrix and the optimizer matrix  

    for (weight_matrix,
         optimizer_matrix,
         tf_weight_matrix,
         tf_optimizer_matrix) in zip(
         weights,
         optimizer_weights,
         weight_matrices_tf,
         optimizer_matrices_tf):

        mask =  mask_for_big_values(weight_matrix,pruning_percent)
        with classifier.graph.as_default():
            # update weights
            classifier.sess.run(tf_weight_matrix.assign(weight_matrix * mask))
            # and corresponding optimizer matrix
            classifier.sess.run(tf_optimizer_matrix.assign(optimizer_matrix * mask))

    # now, lets look on weights distribution (zero values are excluded)
    weights = classifier.sess.run(weight_matrices_tf)
    plot_histogram(weights,'weights_distribution_after_pruning',include_zeros=False)

    accuracy, loss = classifier.evaluate(data_provider=test_data_provider,
                                         batch_size=ConfigNetworkDensePruned.batch_size)
    print('Accuracy on test before fine-tuning: {accuracy}, loss on test: {loss}'.format(
                                                        accuracy=accuracy, loss=loss))

    # fine-tune classifier 
    classifier.fit(n_epochs=5,
                   batch_size=ConfigNetworkDensePruned.batch_size,
                   learning_rate_schedule=ConfigNetworkDensePruned.learning_rate_schedule,
                   train_data_provider=train_data_provider,
                   validation_data_provider=validation_data_provider,
                   test_data_provider=test_data_provider)

    # plot weights distribution again to see the difference
    weights = classifier.sess.run(weight_matrices_tf)
    plot_histogram(weights,'weights_distribution_after_fine_tuning',include_zeros=False)
weightPruning(0)

# %%
weightPruning(25)

# %%
weightPruning(50)

# %%
#60, 70, 80, 90, 95, 97, 99
weightPruning(60)

# %%
weightPruning(70)

# %%
weightPruning(80)

# %%
weightPruning(90)

# %%
weightPruning(95)

# %%
weightPruning(97)

# %%
weightPruning(99)

# %%
class FullyConnectedClassifierSparse(FullyConnectedClassifier):

    def __init__(self,
                 input_size: int,
                 n_classes: int,
                 sparse_layers: list,
                 model_path: str,
                 activation_fn=tf.nn.relu,
                 scope='FullyConnectedClassifierSparse',
                 verbose=True):

        self.input_size = input_size
        self.n_classes = n_classes
        self.sparse_layers = sparse_layers
        self.model_path = model_path
        self.activation_fn = activation_fn
        self.scope = scope
        self.verbose = verbose

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(self.scope):

                self._create_placeholders()

                self.logits = self._build_network(inputs=self.inputs,
                                                  sparse_layers=self.sparse_layers,
                                                  activation_fn=self.activation_fn)

                self.loss = self._create_loss(logits=self.logits,
                                              labels=self.labels)

                self._create_metrics(logits=self.logits,
                                     labels=self.labels,
                                     loss=self.loss)

                self.saver = self._create_saver(tf.global_variables())
                self.init_variables(tf.global_variables())

                if self.verbose:
                    print('\nSuccessfully created graph for {model}.'.format(
                                                            model=self.scope))
                    print('Number of parameters (four bytes == 1 parameter): {}.\n'.format(
                         calculate_number_of_sparse_parameters(
                                                            self.sparse_layers)))

    def _create_placeholders(self):
    
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=(None, self.input_size),
                                     name='inputs')
    
        self.labels = tf.placeholder(dtype=tf.int64,
                                     shape=None,
                                     name='labels')

        # for compatibility with dense model
        self.keep_prob = tf.placeholder(dtype=tf.float32,
                                        shape=(),
                                        name='keep_prob')

    def _build_network(self,
                       inputs: tf.Tensor,
                       sparse_layers: list,
                       activation_fn: callable) -> tf.Tensor:
    
        with tf.variable_scope('network'):
    
            net = inputs
    
            self.weight_tensors = []

            bias_initializer = tf.constant_initializer(0.1)

            for i, layer in enumerate(sparse_layers):
    
                with tf.variable_scope('layer_{layer}'.format(layer=i+1)):

                    # create variables based on sparse values                    
                    with tf.variable_scope('sparse'):

                        indicies = tf.get_variable(name='indicies',
                                                   initializer=layer.indices,
                                                   dtype=tf.int16)

                        values = tf.get_variable(name='values',
                                                 initializer=layer.values,
                                                 dtype=tf.float32)

                        dense_shape = tf.get_variable(name='dense_shape',
                                                      initializer=layer.dense_shape,
                                                      dtype=tf.int64)

                    # create a weight tensor based on the created variables
                    weights = tf.sparse_to_dense(tf.cast(indicies, tf.int64),
                                                 dense_shape,
                                                 values)

                    self.weight_tensors.append(weights)
        
                    name = 'bias'
                    bias = tf.get_variable(name=name,
                                           initializer=layer.bias)
    
                    net = tf.matmul(net, weights) + bias
    
                    if i < len(sparse_layers) - 1:
                        net = activation_fn(net)
    
            return net

    def _create_loss(self,
                     logits: tf.Tensor,
                     labels: tf.Tensor) -> tf.Tensor:
    
        with tf.variable_scope('loss'):
            classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            logits=logits, labels=labels,
                                            name='classification_loss')
    
            classification_loss = tf.reduce_mean(classification_loss,
                                                 name='classification_loss_averaged')

        return classification_loss

# %% [markdown]
# # Unit/Neuron Pruning

# %%
def get_sparse_values_indices_k(weights,k):
    k_len = len(weights)
    k_len = int(k_len * k)
    weights[0:k_len] = 0
    values = weights[weights != 0]
    indices = np.transpose(np.nonzero(weights))
    return values, indices

# %%
def PruningExample(k):
    classifier = FullyConnectedClassifier(
                                input_size=ConfigNetworkDensePruned.input_size,
                                n_classes=ConfigNetworkDensePruned.n_classes,
                                layer_sizes=ConfigNetworkDensePruned.layer_sizes,
                                model_path=ConfigNetworkDensePruned.model_path,
                                dropout=ConfigNetworkDensePruned.dropout,
                                weight_decay=ConfigNetworkDensePruned.weight_decay,
                                activation_fn=ConfigNetworkDensePruned.activation_fn,
                                pruning_threshold=ConfigNetworkDensePruned.pruning_threshold)
    # restore a model
    classifier.load_model()

    accuracy, loss = classifier.evaluate(data_provider=test_data_provider,
                                         batch_size=ConfigNetworkDensePruned.batch_size)
    print('Accuracy on test with dense model (pruned): {accuracy}, loss on test: {loss}'.format(
                                                       accuracy=accuracy, loss=loss))

    weight_matrices, biases = classifier.sess.run([classifier.weight_matrices,
                                                   classifier.biases])
    sparse_layers = []
    # turn dense pruned weights into sparse indices and values
    for weights, bias in zip(weight_matrices, biases):

        if(len(weights[0])!=10):
            values, indices =  get_sparse_values_indices_k(weights,k)
        else:
            values, indices =  get_sparse_values_indices_k(weights,0)
        shape = np.array(weights.shape).astype(np.int64)
        sparse_layers.append( SparseLayer(values=values.astype(np.float32),
                                                       indices=indices.astype(np.int16),
                                                       dense_shape=shape,
                                                       bias=bias))

    # create sparse classifier
    sparse_classifier = FullyConnectedClassifierSparse(
                                input_size=ConfigNetworkSparse.input_size,
                                n_classes=ConfigNetworkSparse.n_classes,
                                sparse_layers=sparse_layers,
                                model_path=ConfigNetworkSparse.model_path,
                                activation_fn=ConfigNetworkSparse.activation_fn)

    # test sparse classifier
    accuracy, loss = sparse_classifier.evaluate(data_provider=test_data_provider,
                                                batch_size=ConfigNetworkSparse.batch_size)
    print('Accuracy on test with sparse model: {accuracy}, loss on test: {loss}'.format(
                                                       accuracy=accuracy, loss=loss))

    # finally, save a sparse model
    sparse_classifier.save_model()
PruningExample(0)

# %% [markdown]
# # For k@ 25%

# %%
PruningExample(0.25)

# %% [markdown]
# # For @50, 60, 70, 80, 90, 95, 97, 99%

# %%
PruningExample(0.60)

# %%
PruningExample(0.70)

# %%
PruningExample(0.80)

# %%
PruningExample(0.90)

# %%
PruningExample(0.95)

# %%
PruningExample(0.97)

# %%
PruningExample(0.99)

# %%
PruningExample(0.50)

# %%

