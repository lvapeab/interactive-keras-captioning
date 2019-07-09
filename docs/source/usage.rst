#####
Usage
#####

********
Training
********

1) Set a training configuration in the config.py_ script. Each parameter is commented. See the `documentation file`_ for further info about each specific hyperparameter. You can also specify the parameters when calling the `main.py`_ script following the syntax `Key=Value`

2) Train!::

    python main.py

********
Decoding
********
Once we have our model trained, we can translate new text using the `caption.py`_ script. If we want to use the models from the first three epochs to translate the `test` split from our dataset, just run::

    python caption.py --models trained_models/tutorial_model/epoch_1 \
                                       trained_models/tutorial_model/epoch_2 \
                              --dataset datasets/Dataset_tutorial_dataset.pkl \
                              --split test

.. _documentation file: https://github.com/lvapeab/interactive-keras-captioning/blob/master/examples/documentation/config.md
.. _config.py: https://github.com/lvapeab/interactive-keras-captioning/blob/master/config.py
.. _main.py: https://github.com/lvapeab/interactive-keras-captioning/blob/master/main.py
.. _caption.py: https://github.com/lvapeab/interactive-keras-captioning/blob/master/caption.py

