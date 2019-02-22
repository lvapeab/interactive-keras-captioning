from keras_wrapper.dataset import Dataset, saveDataset, loadDataset

import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S')


def update_dataset_from_file(ds,
                             input_text_filename,
                             params,
                             splits=None,
                             output_text_filename=None,
                             remove_outputs=False,
                             compute_state_below=False,
                             recompute_references=False):
    """
    Updates the dataset instance from a text file according to the given params.
    Used for sampling

    :param ds: Dataset instance
    :param input_text_filename: New inputs.
    :param params: Parameters for building the dataset
    :param splits: Splits to sample
    :param output_text_filename: New output sentences
    :param remove_outputs: Remove outputs from dataset (if True, will ignore the output_text_filename parameter)
    :param compute_state_below: Compute state below input (shifted target text for professor teaching)
    :param recompute_references: Whether we should rebuild the references of the dataset or not.

    :return: Dataset object with the processed data
    """

    logging.info("<<< Updating Dataset instance " + ds.name + " ... >>>")

    if splits is None:
        splits = ['val']

    if output_text_filename is None:
        recompute_references = False

    for split in splits:
        if split == 'train':
            output_type = params.get('OUTPUTS_TYPES_DATASET', ['dense-text'] if 'sparse' in params['LOSS'] else ['text'])[0]
        else:
            # Type of val/test outuput is always 'text' or 'dense-text'
            output_type = 'dense-text' if 'sparse' in params['LOSS'] else 'text'

        if remove_outputs:
            ds.removeOutput(split,
                            id=params['OUTPUTS_IDS_DATASET'][0])
            recompute_references = False

        elif output_text_filename is not None:
            ds.setOutput(output_text_filename,
                         split,
                         type=output_type,
                         id=params['OUTPUTS_IDS_DATASET'][0],
                         tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                         build_vocabulary=False,
                         pad_on_batch=params.get('PAD_ON_BATCH', True),
                         fill=params.get('FILL', 'end'),
                         sample_weights=params.get('SAMPLE_WEIGHTS', True),
                         max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 100),
                         max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                         min_occ=params.get('MIN_OCCURRENCES_OUTPUT_VOCAB', 0),
                         bpe_codes=params.get('BPE_CODES_PATH', None),
                         label_smoothing=params.get('LABEL_SMOOTHING', 0.),
                         overwrite_split=True)

        # INPUT DATA
        ds.setInput(input_text_filename,
                    split,
                    type=params.get('INPUTS_TYPES_DATASET', ['text', 'text'])[0],
                    id=params['INPUTS_IDS_DATASET'][0],
                    tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                    build_vocabulary=False,
                    pad_on_batch=params.get('PAD_ON_BATCH', True),
                    fill=params.get('FILL', 'end'),
                    max_text_len=params.get('MAX_INPUT_TEXT_LEN', 100),
                    max_words=params.get('INPUT_VOCABULARY_SIZE', 0),
                    min_occ=params.get('MIN_OCCURRENCES_INPUT_VOCAB', 0),
                    bpe_codes=params.get('BPE_CODES_PATH', None),
                    overwrite_split=True)
        if compute_state_below and output_text_filename is not None:
            # INPUT DATA
            ds.setInput(output_text_filename,
                        split,
                        type=params.get('INPUTS_TYPES_DATASET', ['text', 'text'])[1],
                        id=params['INPUTS_IDS_DATASET'][1],
                        pad_on_batch=params.get('PAD_ON_BATCH', True),
                        tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                        build_vocabulary=False,
                        offset=1,
                        fill=params.get('FILL', 'end'),
                        max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 100),
                        max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                        min_occ=params.get('MIN_OCCURRENCES_OUTPUT_VOCAB', 0),
                        bpe_codes=params.get('BPE_CODES_PATH', None),
                        overwrite_split=True)
        else:
            ds.setInput(None,
                        split,
                        type='ghost',
                        id=params['INPUTS_IDS_DATASET'][-1],
                        required=False,
                        overwrite_split=True)

        if params['ALIGN_FROM_RAW']:
            ds.setRawInput(input_text_filename,
                           split,
                           type='file-name',
                           id='raw_' + params['INPUTS_IDS_DATASET'][0],
                           overwrite_split=True)

        # If we had multiple references per sentence
        if recompute_references:
            keep_n_captions(ds, repeat=1, n=1, set_names=params['EVAL_ON_SETS'])

    return ds


def build_dataset(params):
    if params['REBUILD_DATASET']:  # We build a new dataset instance
        if (params['VERBOSE'] > 0):
            silence = False
            logging.info('Building ' + params['DATASET_NAME'] + ' dataset')
        else:
            silence = True

        base_path = params['DATA_ROOT_PATH']
        name = params['DATASET_NAME']
        ds = Dataset(name, base_path, silence=silence)

        # OUTPUT DATA
        # Let's load the train, val and test splits of the descriptions (outputs)
        #    the files include a description per line. In this dataset a variable number
        #    of descriptions per video are provided.
        ds.setOutput(base_path + '/' + params['DESCRIPTION_FILES']['train'],
                     'train',
                     type=params['OUTPUTS_TYPES_DATASET'][0],
                     id=params['OUTPUTS_IDS_DATASET'][0],
                     build_vocabulary=True,
                     tokenization=params['TOKENIZATION_METHOD'],
                     fill=params['FILL'],
                     pad_on_batch=True,
                     max_text_len=params['MAX_OUTPUT_TEXT_LEN'],
                     sample_weights=params['SAMPLE_WEIGHTS'],
                     min_occ=params['MIN_OCCURRENCES_OUTPUT_VOCAB'])

        ds.setOutput(base_path + '/' + params['DESCRIPTION_FILES']['val'],
                     'val',
                     type=params['OUTPUTS_TYPES_DATASET'][0],
                     id=params['OUTPUTS_IDS_DATASET'][0],
                     build_vocabulary=True,
                     pad_on_batch=True,
                     tokenization=params['TOKENIZATION_METHOD'],
                     sample_weights=params['SAMPLE_WEIGHTS'],
                     max_text_len=params['MAX_OUTPUT_TEXT_LEN_TEST'],
                     min_occ=params['MIN_OCCURRENCES_OUTPUT_VOCAB'])

        ds.setOutput(base_path + '/' + params['DESCRIPTION_FILES']['test'],
                     'test',
                     type=params['OUTPUTS_TYPES_DATASET'][0],
                     id=params['OUTPUTS_IDS_DATASET'][0],
                     build_vocabulary=True,
                     pad_on_batch=True,
                     tokenization=params['TOKENIZATION_METHOD'],
                     sample_weights=params['SAMPLE_WEIGHTS'],
                     max_text_len=params['MAX_OUTPUT_TEXT_LEN_TEST'],
                     min_occ=params['MIN_OCCURRENCES_OUTPUT_VOCAB'])

        # INPUT DATA
        # Let's load the associated videos (inputs)
        # we must take into account that in this dataset we have a different number of sentences per video,
        # for this reason we introduce the parameter 'repeat_set'=num_captions, where num_captions is a list
        # containing the number of captions in each video.

        num_captions_train = np.load(base_path + '/' + params['DESCRIPTION_COUNTS_FILES']['train'])
        num_captions_val = np.load(base_path + '/' + params['DESCRIPTION_COUNTS_FILES']['val'])
        num_captions_test = np.load(base_path + '/' + params['DESCRIPTION_COUNTS_FILES']['test'])

        for n_feat, feat_type in enumerate(params['FEATURE_NAMES']):
            for split, num_cap in zip(['train', 'val', 'test'],
                                      [num_captions_train, num_captions_val,
                                       num_captions_test]):
                list_files = base_path + '/' + params['FRAMES_LIST_FILES'][split] % feat_type
                counts_files = base_path + '/' + params['FRAMES_COUNTS_FILES'][split] % feat_type

                ds.setInput([list_files, counts_files],
                            split,
                            type=params['INPUTS_TYPES_DATASET'][n_feat],
                            id=params['INPUTS_IDS_DATASET'][0],
                            repeat_set=num_cap,
                            max_video_len=params['NUM_FRAMES'],
                            feat_len=params['IMG_FEAT_SIZE'])

        if len(params['INPUTS_IDS_DATASET']) > 1:
            ds.setInput(base_path + '/' + params['DESCRIPTION_FILES']['train'],
                        'train',
                        type=params['INPUTS_TYPES_DATASET'][-1],
                        id=params['INPUTS_IDS_DATASET'][-1],
                        required=False,
                        tokenization=params['TOKENIZATION_METHOD'],
                        pad_on_batch=True,
                        build_vocabulary=params['OUTPUTS_IDS_DATASET'][0],
                        offset=1,
                        fill=params['FILL'],
                        max_text_len=params['MAX_OUTPUT_TEXT_LEN'],
                        max_words=params['OUTPUT_VOCABULARY_SIZE'],
                        min_occ=params['MIN_OCCURRENCES_OUTPUT_VOCAB'])

            ds.setInput(None, 'val', type='ghost',
                        id=params['INPUTS_IDS_DATASET'][-1], required=False)
            ds.setInput(None, 'test', type='ghost',
                        id=params['INPUTS_IDS_DATASET'][-1], required=False)

        # Process dataset for keeping only one caption per video and storing the rest in a dict() with the following format:
        #        ds.extra_variables[set_name][id_output][img_position] = [cap1, cap2, cap3, ..., capN]
        keep_n_captions(ds, repeat=[num_captions_val, num_captions_test], n=1,
                        set_names=['val', 'test'])

        # We have finished loading the dataset, now we can store it for using it in the future
        saveDataset(ds, params['DATASET_STORE_PATH'])
    else:
        # We can easily recover it with a single line
        ds = loadDataset(params['DATASET_STORE_PATH'] + '/Dataset_' + params[
            'DATASET_NAME'] + '.pkl')

    return ds



def keep_n_captions(ds, repeat, n=1, set_names=None):
    """
    Keeps only n captions per image and stores the rest in dictionaries for a later evaluation
    :param ds: Dataset object
    :param repeat: Number of input samples per output
    :param n: Number of outputs to keep.
    :param set_names: Set name.
    :return:
    """

    if set_names is None:
        set_names = ['val', 'test']

    for s, r in zip(set_names, repeat):
        logging.info('Keeping ' + str(n) + ' captions per input on the ' + str(s) + ' set.')

        ds.extra_variables[s] = dict()
        n_samples = getattr(ds, 'len_' + s)
        # Process inputs
        for id_in in ds.ids_inputs:
            new_X = []
            if id_in in ds.optional_inputs:
                try:
                    X = getattr(ds, 'X_' + s)
                    i = 0
                    for next_repeat in r:
                        for j in range(n):
                            new_X.append(X[id_in][i+j])
                    i += next_repeat
                    setattr(ds, 'X_' + s + '[' + id_in + ']', new_X)
                except Exception:
                    pass
            else:
                X = getattr(ds, 'X_' + s)
                i = 0
                for next_repeat in r:
                    for j in range(n):
                        new_X.append(X[id_in][i + j])
                    i += next_repeat
                aux_list = getattr(ds, 'X_' + s)
                aux_list[id_in] = new_X
                setattr(ds, 'X_' + s, aux_list)
                del aux_list
        # Process outputs
        for id_out in ds.ids_outputs:
            new_Y = []
            Y = getattr(ds, 'Y_' + s)
            dict_Y = dict()
            count_samples = 0
            i = 0
            for next_repeat in r:
                dict_Y[count_samples] = []
                for j in range(next_repeat):
                    if j < n:
                        new_Y.append(Y[id_out][i + j])
                    dict_Y[count_samples].append(Y[id_out][i + j])
                count_samples += 1
                i += next_repeat
            aux_list = getattr(ds, 'Y_' + s)
            aux_list[id_out] = new_Y
            setattr(ds, 'Y_' + s, aux_list)
            del aux_list

            # store dictionary with img_pos -> [cap1, cap2, cap3, ..., capN]
            ds.extra_variables[s][id_out] = dict_Y

        new_len = len(new_Y)
        setattr(ds, 'len_' + s, new_len)

        logging.info('Samples reduced to ' + str(new_len) + ' in ' + s + ' set.')
