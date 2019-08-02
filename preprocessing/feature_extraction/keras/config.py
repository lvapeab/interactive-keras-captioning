def load_parameters():
    """
        Loads the defined parameters
    """
    # Input data params
    TASK_NAME = 'Flickr8k'
    DATA_ROOT_PATH = '/home/lvapeab/DATASETS/' + TASK_NAME + '/'

    # Image and features files (the chars {} will be replaced by each type of features)
    IMG_FILES = {'train': 'Annotations/train_list_images.txt',
                 'val': 'Annotations/val_list_images.txt',
                 'test': 'Annotations/test_list_images.txt'
                 }

    EXTRACT_ON_SETS = ['train', 'val', 'test']  # Possible values: 'train', 'val' and 'test' (external evaluator)

    # Feature extractor model. See keras_applications for the supported models.
    # By default, we support 'InceptionV3', 'ResNet152' and NASNetLarge; but adding other models is trivial.
    MODEL_TYPE = 'NASNetLarge'
    SPATIAL_LAST = True  # Keras puts the spatial dimensions at the start (e.g. (8, 8, 2048). We may want to put them at the end (2048, 8, 8)

    # Results plot and models storing parameters
    EXTRA_NAME = ''  # This will be appended to the end of the model name
    MODEL_NAME = TASK_NAME + '_' + MODEL_TYPE
    MODEL_NAME += EXTRA_NAME

    STORE_PATH = DATA_ROOT_PATH + '/Features/' + MODEL_NAME + '/'  # Models and evaluation results will be stored here

    SPLIT_OUTPUT = True

    VERBOSE = 1  # Verbosity level
    # ============================================
    parameters = locals().copy()
    return parameters
