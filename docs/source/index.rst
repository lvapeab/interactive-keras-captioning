============================
Interactive Keras Captioning
============================
Interactive multimedia captioning with Keras (Theano and Tensorflow). Given an input image or video, we describe its content.

`Checkout the live demo!`_

.. image:: ../rnn_model.png
   :width: 100 %
   :alt: alternate text
   :align: left

|
|

.. image:: ../transformer_model.png
   :width: 100 %
   :alt: alternate text
   :align: left


Interactive captioning
**********************
Interactive-predictive pattern recognition is a collaborative human-machine framework for obtaining high-quality predictions while minimizing the human effort spent during the process.

It consists in an iterative prediction-correction process: each time the user introduces a correction to a hypothesis, the system reacts offering an alternative, considering the user feedback.

For further reading about this framework, please refer to `Interactive Neural Machine Translation`_, `Online Learning for Effort Reduction in Interactive Neural Machine Translation`_ and `Interactive-predictive neural multimodal systems`_.

********
Features
********
 * Attention-based RNN and Transformer models.
 * Support for GRU/LSTM networks:
   - Regular GRU/LSTM units.
   - Conditional_ GRU/LSTM units in the decoder.
   - Multilayered residual GRU/LSTM networks.
 * Attention model over the input sequence of annotations.
   - Supporting Bahdanau (Add) and Luong (Dot) attention mechanisms.
   - Also supports double stochastic attention.
 * Peeked decoder: The previously generated word is an input of the current timestep.
 * Beam search decoding.
   - Featuring length and source coverage normalization.
 * Ensemble decoding.
 * Caption scoring.
 * N-best list generation (as byproduct of the beam search process).
 * Use of pretrained (Glove_ or Word2Vec_) word embedding vectors.
 * MLPs for initializing the RNN hidden and memory state.
 * Spearmint_ wrapper for hyperparameter optimization.
 * Client-server_ architecture for web demos.

.. _Spearmint: https://github.com/HIPS/Spearmint
.. _Glove: http://nlp.stanford.edu/projects/glove/
.. _Conditional: https://arxiv.org/abs/1703.04357
.. _Word2Vec: https://code.google.com/archive/p/word2vec/
.. _Client-server: https://github.com/lvapeab/interactive-keras-captioning/tree/master/demo-web
.. _Checkout the live demo!: http://casmacat.prhlt.upv.es/interactive-seq2seq/
.. _Interactive Neural Machine Translation: https://www.sciencedirect.com/science/article/pii/S0885230816301000
.. _Online Learning for Effort Reduction in Interactive Neural Machine Translation: https://arxiv.org/abs/1802.03594
.. _Interactive-predictive neural multimodal systems: https://arxiv.org/abs/1905.12980


*****
Guide
*****
.. toctree::
   :maxdepth: 2

   requirements
   usage
   configuration
   modules
   help


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
