#####################
Configuration options
#####################

This document describes the available hyperparameters used for training Interactive Keras Captioning.

These hyperparameters are set in the `config.py`_ script or via command-line-interface.

Warning! This document may be out-dated. Please, check `config.py`_ for new hyperparameters.



Naming and experiment setup
===========================
  * **TASK_NAME**: Task name. Used for naming and for indexing files.
  * **DATASET_NAME**: Dataset name. Used for naming.
  * **DATA_ROOT_PATH**: Path to the data
  * **TRG_LAN**: Language of the target text. Used for naming and for coputing language-dependent metrics (e.g. Meteor)
  * **INPUT_DATA_TYPE**:  Type of the input data. Supported: 'image-features', 'raw-image', 'video-features', 'raw-video'. raw-* somewhat untested.




Options for input data type '*-features'
========================================
   * **FEATURE_NAMES**:  Feature names.
   * **FEATURE_DIMENSION**:  Dimension of the features. List of shapes or integer specifying the last dimension.
   * **NUM_FRAMES**:  Number of frames per sample (must be set to -1 in the case of image-features).
   * **FRAMES_LIST_FILES**: Path to the image and feature files (the chars {} will be replaced by each type of features)
   * **FRAMES_COUNTS_FILES**: Frame counts for each sample (leave blank for image-features).




Options for input data type 'raw-image'
=======================================

   * **IMG_SIZE**:  Size of the input images (will be resized to the desired size).
   * **IMG_CROP_SIZE**: Size of the image crops inputted to the model
   * **FEATURE_NAMES**: Name of the features computed by the model.
   * **NORMALIZE**: Apply image normalization.
   * **NORMALIZATION_TYPE**: Type of normalization.
   * **DATA_AUGMENTATION**: Apply data augmentation on input data .
   * **FRAMES_LIST_FILES**: Path to the image and feature files (the chars {} will be replaced by each type of features)


Output data
===========
   * **DESCRIPTION_FILES**: Path to the files containing the captions
   * **LABELS_PER_SAMPLE**: Number of references per sample. Set to 0 for using a variable number of references per sample.
   * **DESCRIPTION_COUNTS_FILES**: If LABELS_PER_SAMPLE == 0, counters of the number of labels for each sample.


Input/output mappings
=====================
  * **INPUTS_IDS_DATASET**: Name of the inputs of the Dataset class.
  * **OUTPUTS_IDS_DATASET**: Name of the outputs of the Dataset class.
  * **INPUTS_IDS_MODEL**: Name of the inputs of the Model.
  * **OUTPUTS_IDS_MODEL**: Name of the outputs of the Model.

Evaluation
==========
  * **METRICS**: List of metric used for evaluating the model. The `coco`_ package is recommended.
  * **EVAL_ON_SETS**: List of splits ('train', 'val', 'test') to evaluate with the metrics from METRICS. Typically: 'val'
  * **EVAL_ON_SETS_KERAS**: List of splits ('train', 'val', 'test') to evaluate with the Keras metrics.
  * **START_EVAL_ON_EPOCH**: The evaluation starts at this epoch.
  * **EVAL_EACH_EPOCHS**: Whether the evaluation frequency units are epochs or updates.
  * **EVAL_EACH**: Evaluation frequency.

Decoding
========
  * **SAMPLING**: Decoding mode. Only 'max_likelihood' tested.
  * **TEMPERATURE**: Multinomial sampling temerature.
  * **BEAM_SEARCH**: Switches on-off the beam search.
  * **BEAM_SIZE**: Beam size.
  * **OPTIMIZED_SEARCH**: Encode the source only once per sample (recommended).


Search normalization
====================
  * **SEARCH_PRUNING**: Apply pruning strategies to the beam search method. It will likely increase decoding speed, but decrease quality.
  * **LENGTH_PENALTY**: Apply length penalty (`Wu et al. (2016)`_).
  * **LENGTH_NORM_FACTOR**: Length penalty factor (`Wu et al. (2016)`_).
  * **NORMALIZE_SAMPLING**: Alternative (simple) length normalization. Normalize hypotheses scores according to their length.
  * **ALPHA_FACTOR**: Normalization according to \|h|**ALPHA_FACTOR (`Wu et al. (2016)`_).

Sampling
========
  * **SAMPLE_ON_SETS**: Splits from where we'll sample.
  * **N_SAMPLES**: Number of samples generated
  * **START_SAMPLING_ON_EPOCH**: First epoch where to start the sampling counter
  * **SAMPLE_EACH_UPDATES**: Sampling frequency (always in #updates)


Word representation
===================
   * **TOKENIZATION_METHOD**: Tokenization applied to the input and output text.
   * **DETOKENIZATION_METHOD**: Detokenization applied to the input and output text.
   * **APPLY_DETOKENIZATION**: Wheter we apply the detokenization method
   * **TOKENIZE_HYPOTHESES**: Whether we tokenize the hypotheses (for computing metrics).
   * **TOKENIZE_REFERENCES**: Whether we tokenize the references (for computing metrics).
   * **BPE_CODES_PATH**: If `TOKENIZATION_METHOD == 'tokenize_bpe'`, sets the path to the learned BPE codes.

Text representation
===================
   * **FILL**: Padding mode: Insert zeroes at the 'start', 'center' or 'end'.
   * **PAD_ON_BATCH**: Make batches of a fixed number of timesteps or pad to the maximum length of the minibatch.

Output text
===========
   * **INPUT_VOCABULARY_SIZE**: Output vocabulary size. Set to 0 for using all, otherwise it will be truncated to these most frequent words.
   * **MIN_OCCURRENCES_OUTPUT_VOCAB**: Discard all output words with a frequency below this threshold.
   * **MAX_INPUT_TEXT_LEN**: Maximum length of the output sentences.
   * **MAX_OUTPUT_TEXT_LEN_TEST**: Maximum length of the output sequence during test time.

Optimization
============
   * **LOSS**: Loss function to optimize.
   * **CLASSIFIER_ACTIVATION**: Last layer activation function.
   * **SAMPLE_WEIGHTS**: Apply a mask to the output sequence. Should be set to True.
   * **LR_DECAY**: Reduce the learning rate each this number of epochs. Set to None if don't want to decay the learning rate
   * **LR_GAMMA**: Decay rate.
   * **LABEL_SMOOTHING**: Epsilon value for label smoothing. Only valid for 'categorical_crossentropy' loss. See [1512.00567](arxiv.org/abs/1512.00567).

Optimizer setup
---------------
   * **OPTIMIZER**: Optimizer to use. See the `available Keras optimizers`_.
   * **LR**: Learning rate.
   * **CLIP_C**: During training, clip L2 norm of gradients to this value.
   * **CLIP_V**: During training, clip absolute value of gradients to this value.
   * **USE_TF_OPTIMIZER**: Use native Tensorflow's optimizer (only for the Tensorflow backend).

Advanced parameters for optimizers
----------------------------------
   * **MOMENTUM**: Momentum value (for SGD optimizer).
   * **NESTEROV_MOMENTUM**: Use Nesterov momentum (for SGD optimizer).
   * **RHO**:  Rho value (for Adadelta and RMSprop optimizers).
   * **BETA_1**:  Beta 1 value (for Adam, Adamax Nadam optimizers).
   * **BETA_2**:  Beta 2 value (for Adam, Adamax Nadam optimizers).
   * **EPSILON**:  Oprimizers epsilon value.

Learning rate schedule
======================
   * **LR_DECAY**: Frequency (number of epochs or updates) between LR annealings. Set to None for not decay the learning rate.
   * **LR_GAMMA**: Multiplier used for decreasing the LR.
   * **LR_REDUCE_EACH_EPOCHS**: Reduce each LR_DECAY number of epochs or updates.
   * **LR_START_REDUCTION_ON_EPOCH**:  Epoch to start the reduction.
   * **LR_REDUCER_TYPE**: Function to reduce. 'linear' and 'exponential' implemented.
   * **LR_REDUCER_EXP_BASE**: Base for the exponential decay.
   * **LR_HALF_LIFE**: Factor/warmup steps for exponenital/noam decay.
   * **WARMUP_EXP**: Warmup steps for noam decay.

Training options
================
   * **MAX_EPOCH**: Stop when computed this number of epochs.
   * **BATCH_SIZE**: Size of each minibatch.
   * **HOMOGENEOUS_BATCHES**: If activated, use batches with similar output lengths, in order to better profit parallel computations.
   * **JOINT_BATCHES**: When using homogeneous batches, size of the maxibatch.
   * **PARALLEL_LOADERS**: Parallel CPU data batch loaders.
   * **EPOCHS_FOR_SAVE**: Save model each this number of epochs.
   * **WRITE_VALID_SAMPLES**: Write validation samples in file.
   * **SAVE_EACH_EVALUATION**: Save the model each time we evaluate.

Early stop
==========
   * **EARLY_STOP** = Turns on/off the early stop regularizer.
   * **PATIENCE**: We'll stop if we don't improve after this number of evaluations
   * **STOP_METRIC**: Stopping metric.

Model main hyperparameters
==========================
   * **MODEL_TYPE**: Model to train. See the `model zoo`_ for the supported architectures.
   * **RNN_TYPE**: RNN unit type ('LSTM' and 'GRU' supported).
   * **INIT_FUNCTION**: Initialization function for matrices (see `keras/initializations`_).
   * **INNER_INIT**: Initialization function for inner RNN matrices.
   * **INIT_ATT**: Initialization function for attention mechism matrices.
   * **TRAINABLE_ENCODER**: Whether the encoder's weights should be modified during training.
   * **TRAINABLE_DECODER**: Whether the decoder's weights should be modified during training.

Embedding options
-----------------
   * **TARGET_TEXT_EMBEDDING_SIZE**: Source language word embedding size.
   * **TRG_PRETRAINED_VECTORS**: Path to target pretrained vectors. See the utils_ folder for preprocessing scripts. Set to None if you don't want to use source pretrained vectors. When using pretrained word embeddings. this parameter must match with the target word embeddings size
   * **TRG_PRETRAINED_VECTORS_TRAINABLE**: Finetune or not the target word embedding vectors.
   * **SCALE_TARGET_WORD_EMBEDDINGS**: Scale target word embeddings by Sqrt(TARGET_TEXT_EMBEDDING_SIZE).
   * **SCALE_SOURCE_WORD_EMBEDDINGS**: Scale source word embeddings by Sqrt(SOURCE_TEXT_EMBEDDING_SIZE).

Deepness
--------
   * **N_LAYERS_ENCODER**: Stack this number of encoding layers.
   * **N_LAYERS_DECODER**: Stack this number of decoding layers.
   * **IMG_EMBEDDING_LAYERS**: Fully-conected layers for visual embedding.
   * **DEEP_OUTPUT_LAYERS**: Additional Fully-Connected layers applied before softmax.

AttentionRNNEncoderDecoder model
================================
   * **ENCODER_RNN_TYPE**: Encoder's RNN unit type ('LSTM' and 'GRU' supported).
   * **USE_CUDNN**: Use CuDNN's implementation of GRU and LSTM (only for Tensorflow backend).
   * **DECODER_RNN_TYPE**: Decoder's RNN unit type ('LSTM', 'GRU', 'ConditionalLSTM' and 'ConditionalGRU' supported).
   * **ATTENTION_MODE**: Attention mode. 'add' (Bahdanau-style) or 'dot' (Luong-style).

Encoder configuration
---------------------
   * **RNN_ENCODER_HIDDEN_SIZE**: Encoder RNN size.
   * **BIDIRECTIONAL_ENCODER**: Use a bidirectional encoder.
   * **BIDIRECTIONAL_DEEP_ENCODER**: Use bidirectional encoder in all stacked encoding layers
   * **BIDIRECTIONAL_MERGE_MODE**: Merge function for bidirectional layers.

Decoder configuration
---------------------
   * **RNN_DECODER_HIDDEN_SIZE**: Decoder RNN size.
   * **ATTENTION_SIZE**: Decoder RNN size.
   * **ADDITIONAL_OUTPUT_MERGE_MODE**: Merge mode for the `deep output layer`_.
   * **SKIP_VECTORS_HIDDEN_SIZE**: Deep output layer size
   * **INIT_LAYERS**: Initialize the first decoder state with these layers (from the encoder).
   * **SKIP_VECTORS_SHARED_ACTIVATION**: Activation for the skip vectors.

Transformer model
=================
   * **MODEL_SIZE**: Transformer model size (dmodel in de paper).
   * **MULTIHEAD_ATTENTION_ACTIVATION**: Activation the input projections in the Multi-Head Attention blocks.
   * **FF_SIZE**: Size of the feed-forward layers of the Transformer model.
   * **N_HEADS**: Number of parallel attention layers of the Transformer model.


Regularizers
============

Regularization functions
------------------------
   * **REGULARIZATION_FN**: Regularization function. 'L1', 'L2' and 'L1_L2' supported.
   * **WEIGHT_DECAY**: L2 regularization in non-recurrent layers.
   * **RECURRENT_WEIGHT_DECAY**: L2 regularization in recurrent layers
   * **DOUBLE_STOCHASTIC_ATTENTION_REG**: Doubly stochastic attention (Eq. 14 from arXiv:1502.03044).

Dropout
-------
   * **DROPOUT_P**: Percentage of units to drop in non-recurrent layers (0 means no dropout).
   * **RECURRENT_DROPOUT_P**: Percentage of units to drop in recurrent layers(0 means no dropout).
   * **ATTENTION_DROPOUT_P**: Percentage of units to drop in attention layers (0 means no dropout).
   * **ATTENTION_DROPOUT_P**: Percentage of units to drop in attention layers (0 means no dropout).

Gaussian noise
--------------
   * **USE_NOISE**: Apply gaussian noise during training.
   * **NOISE_AMOUNT**: Amount of noise.

Batch normalization
-------------------
   * **USE_BATCH_NORMALIZATION**:  Batch normalization regularization in non-recurrent layers and recurrent inputs. If True it is recommended to deactivate Dropout.
   * **BATCH_NORMALIZATION_MODE**: Sample-wise or feature-wise BN mode.

Additional normalization layers
-------------------------------
   * **USE_PRELU**: Apply use PReLU activations as regularizer.
   * **USE_L1**: L1 normalization on the features.
   * **USE_L2**: Apply L2 function on the features.


Tensorboard
===========
   * **TENSORBOARD**: Switches On/Off the tensorboard callback.
   * **LOG_DIR**: irectory to store teh model. Will be created inside STORE_PATH.
   * **EMBEDDINGS_FREQ**: Frequency (in epochs) at which selected embedding layers will be saved.
   * **EMBEDDINGS_LAYER_NAMES**: A list of names of layers to keep eye on. If None or empty list all the embedding layer will be watched.
   * **EMBEDDINGS_METADATA**: Dictionary which maps layer name to a file name in which metadata for this embedding layer is saved.
   * **LABEL_WORD_EMBEDDINGS_WITH_VOCAB**: Whether to use vocabularies as word embeddings labels (will overwrite EMBEDDINGS_METADATA).
   * **WORD_EMBEDDINGS_LABELS**: Vocabularies for labeling. Must match EMBEDDINGS_LAYER_NAMES.

Storage and plotting
====================
   * **MODEL_NAME**: Name for the model.
   * **EXTRA_NAME**: MODEL_NAME suffix
   * **STORE_PATH**: Models and evaluation results will be stored here.
   * **DATASET_STORE_PATH**: Dataset instance will be stored here.

   * **SAMPLING_SAVE_MODE**: Save evaluation outputs in this format. Set to 'list' for a raw file.
   * **VERBOSE**: Verbosity level.
   * **RELOAD**: Reload a stored model. If 0 start training from scratch, otherwise use the model from this epoch/update.
   * **REBUILD_DATASET**: Build dataset again or use a stored instance.

.. _model zoo: https://github.com/lvapeab/interactive-keras-captioning/blob/master/captioner/model_zoo.py
.. _deep output layer: https://arxiv.org/abs/1312.6026
.. _utils: https://github.com/lvapeab/interactive-keras-captioning/tree/master/utils
.. _keras/initializations: https://github.com/MarcBS/keras/blob/master/keras/initializations.py
.. _available Keras optimizers: https://github.com/MarcBS/keras/blob/master/keras/optimizers.py
.. _Wu et al. (2016): https://arxiv.org/abs/1609.08144
.. _coco: https://github.com/lvapeab/coco-caption
.. _config.py: https://github.com/lvapeab/interactive-keras-captioning/blob/master/config.py