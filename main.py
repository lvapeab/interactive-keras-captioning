import ast
import logging
import sys
import argparse
from timeit import default_timer as timer
from config import load_parameters
from captioner import check_params
from captioner.training import train_model
from captioner.apply_model import sample_ensemble

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Train or apply video captioning")
    parser.add_argument("-ch", "--changes", nargs="*", help="Changes to config, following the syntax Key=Value",
                        default="")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    parameters = load_parameters()
    try:
        for arg in args.changes:
            try:
                k, v = arg.split('=')
            except ValueError:
                print ('Overwritten arguments must have the form key=Value. \n Currently are: %s' % str(args.changes))
                exit(1)
            try:
                parameters[k] = ast.literal_eval(v)
            except ValueError:
                parameters[k] = v
    except ValueError:
        print ('Error processing arguments: (', k, ",", v, ")")
        exit(2)

    check_params(parameters)

    if parameters['MODE'] == 'training':
        logging.info('Running training.')
        train_model(parameters)
        logging.info('Done!')
    elif parameters['MODE'] == 'sampling':
        logging.error('Depecrated function. For sampling from a trained model, please run caption.py.')
        exit(2)
