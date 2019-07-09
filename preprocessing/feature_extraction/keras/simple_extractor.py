from config import load_parameters
import numpy as np
import time
from keras_wrapper.extra.read_write import *
import ast
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

from keras.preprocessing import image
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import preprocess_input as preprocess_input_nasnet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.resnet_v2 import preprocess_input as preprocess_input_resnet152


def nasNetLarge(model, img_path):
    img = image.load_img(img_path, target_size=(331, 331))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_nasnet(x)
    features = model.predict(x)
    return features


def inceptionV3(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_inception(x)
    features = model.predict(x)
    return features


def resNet152(model, img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_resnet152(x)
    features = model.predict(x)
    return features


def apply_Feature_Extractor_model(params):
    """
    Apply a previously trained model.
    :param params: Hyperparameters
    :return:
    """
    model = None
    if params['MODEL_TYPE'] == 'InceptionV3':
        model = InceptionV3(weights='imagenet', include_top=False)
    elif params['MODEL_TYPE'] == 'NASNetLarge':
        model = NASNetLarge(weights='imagenet', include_top=False)
    elif params['MODEL_TYPE'] == 'ResNet152':
        model = ResNet152V2(weights='imagenet', include_top=False)

    print(model.summary())
    base_path = params['DATA_ROOT_PATH']

    for s in params['EXTRACT_ON_SETS']:
        if params['SPLIT_OUTPUT']:
            path_general = params['STORE_PATH'] + '/' + params.get('MODEL_TYPE', 'features') + '/' + s + '/'
            if not os.path.isdir(path_general):  # create dir if it doesn't exist
                os.makedirs(path_general)
        list_filepath = base_path + '/' + params['IMG_FILES'][s]
        image_list = file2list(list_filepath)
        eta = -1
        start_time = time.time()
        n_images = len(image_list)
        for n_sample, imname in list(enumerate(image_list)):
            if params['MODEL_TYPE'] == 'InceptionV3':
                features = inceptionV3(model, imname)
            elif params['MODEL_TYPE'] == 'NASNetLarge':
                features = nasNetLarge(model, imname)
            elif params['MODEL_TYPE'] == 'ResNet152':
                features = resNet152(model, imname)

            # Keras puts the spatial dimensions at the start. We may want to put them at the end
            if params.get('SPATIAL_LAST', True):
                features = features.transpose(0, 3, 1, 2)

            filepath = path_general + imname.split('/')[-1][:-4] + '.npy' if imname.split('/')[-1][-4:] == '.jpg' or imname.split('/')[-1][-4:] == '.png' else path_general + imname.split('/')[
                -1] + '.npy'
            numpy2file(filepath, features, permission='wb', split=False)
            sys.stdout.write('\r')
            sys.stdout.write("\t Processed %d/%d  -  ETA: %ds " % (n_sample, n_images, int(eta)))
            sys.stdout.flush()
            eta = (n_images - n_sample) * (time.time() - start_time) / max(n_sample, 1)
    print("Features saved in", path_general)


if __name__ == "__main__":
    params = load_parameters()
    try:
        for arg in sys.argv[1:]:
            k, v = arg.split('=')
            params[k] = ast.literal_eval(v)
    except:
        print('Overwritten arguments must have the form key=Value')
        exit(1)

    apply_Feature_Extractor_model(params)

    logging.info('Done!')
