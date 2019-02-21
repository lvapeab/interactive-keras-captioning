import pytest
import copy
from config import load_parameters
from data_engine.prepare_data import build_dataset, keep_n_captions
from keras_wrapper.dataset import Dataset, loadDataset


def test_build_datset():
    params = load_parameters()
    for verbose in range(2):
        params['REBUILD_DATASET'] = True
        params['VERBOSE'] = verbose
        params['DATASET_STORE_PATH'] = './'
        ds = build_dataset(params)
        assert isinstance(ds, Dataset)
        len_splits = [('train', 9900), ('val', 100), ('test', 2996)]
        for split, len_split in len_splits:
            assert eval('ds.len_' + split) == len_split
            assert eval('all(ds.loaded_' + split + ')')
            assert len(eval('ds.X_' + split + str([params['INPUTS_IDS_DATASET'][0]]))) == len_split
            assert len(eval('ds.Y_' + split + str([params['OUTPUTS_IDS_DATASET'][0]]))) == len_split


def test_load_dataset():
    params = load_parameters()
    ds = loadDataset('./Dataset_' + params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '.pkl')
    assert isinstance(ds, Dataset)
    assert isinstance(ds.vocabulary, dict)
    assert len(list(ds.vocabulary)) >= 3
    for voc in ds.vocabulary:
        assert len(list(ds.vocabulary[voc])) == 2


def test_keep_n_captions():
    params = load_parameters()
    params['REBUILD_DATASET'] = True
    params['DATASET_STORE_PATH'] = './'
    ds = build_dataset(params)
    len_splits = {'train': 9900, 'val': 100, 'test': 2996}

    for splits in [[], None, ['val'], ['val', 'test']]:
        keep_n_captions(ds, 1, n=1, set_names=splits)
        if splits is not None:
            for split in splits:
                len_split = len_splits[split]
                assert eval('ds.len_' + split) == len_split
                assert eval('all(ds.loaded_' + split + ')')
                assert len(eval('ds.X_' + split + str([params['INPUTS_IDS_DATASET'][0]]))) == len_split
                assert len(eval('ds.Y_' + split + str([params['OUTPUTS_IDS_DATASET'][0]]))) == len_split

    if __name__ == '__main__':
        pytest.main([__file__])
