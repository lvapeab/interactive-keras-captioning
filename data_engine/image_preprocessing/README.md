
## Steps for data pre-processing

1. Run prepare_files.sh inserting the path to your dataset annotations.

2. Alternative 1 - using feature vectors:
    2.1. Copy the generated <set-split>_list.txt files into the same folder where you have the features extracted.
    2.2. Run /utils/prepare_features.py inserting your features path and their identifier (this script must be run once
        for each type of features used.)
    2.3. Run convert_to_lists.py inserting the path to your dataset for applying the final division in lists that must
        be included in ../config.py (this script must be run once for each type of features used.)

2. Alternative 2 - using raw images:
    2.1. Copy the generated <set-split>_list.txt files into the same folder where you have the annotations extracted.
    2.2. Run generate_img_lists.py inserting your images path and their identifier.
    
3. If using queries as additional inputs (keywords for generating keyword-oriented sentences)
    3.1. Run an object detection on the images (LSDA in this case)
    3.2. Run prepare_classes_from_lsda.sh (adapt it to your object detector's needs)
    3.3. Run generate_query_corpus.py
    

Note: all pre-processing files are ready for processing the dataset Flickr8k, some minor changes might be needed for other datasets.
