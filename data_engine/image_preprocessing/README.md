
## Steps for data pre-processing

1. Run `prepare_text_files.sh` inserting the path to your dataset annotations.

2. Process the images. You have two alternatives: using pre-extracted features or using the raw images.
    2.1 Alternative 1 - using feature vectors:
        2.1.1. Set the generated <set-split>_list.txt in the `generate_feature_lists.py`.
        
        2.1.2. Run `generate_feature_lists.py`
        
    2.2. Alternative 2 - using raw images:
        2.2.1. Copy the generated <set-split>_list.txt files into the same folder where you have the annotations extracted.
        2.2.2. Run generate_img_lists.py inserting your images path and their identifier.
    
Note: all pre-processing scripts are ready for processing the dataset Flickr8k, some minor changes might be needed for other datasets.
