# Image preprocessing

Shows how to preprocess the image data for the Flickr8k dataset. Minor changes might be needed for other datasets.

## Structure

We assume that our dataset will be stored in the directory `~/DATASETS/Flickr8k`.

We will organize our dataset in 3 folders:

1. `Annotations`: Contains all the non-image related information. E.g. Captions, labels, lists containing image/feature paths for each split, etc.

2. `Images`: Contains the raw images from the dataset.

3. `Features`: Contains the features extracted for the images.

Following this structure, let's organize our folders:

1. Create the directories:   ``cd ~/DATASETS/Flickr8k; mkdir Features Images Annotations``
      
2. Extract the `Flickr8k` images in `Images` and put the rest of files in `Annotations`.

3. Rename some files from `Annotations`:
      ```
      cd Annotations;
      mv Flickr_8k.trainImages.txt train_list.txt;
      mv Flickr_8k.devImages.txt val_list.txt; 
      mv Flickr_8k.testImages.txt test_list.txt;
      ```

## Data pre-processing

`cd` to this folder (`cd preprocessing/image_preprocessing/`


1. Run `prepare_text_files.sh` inserting the path to your dataset annotations:
      ```
      bash prepare_text_files.sh ~/DATASETS/Flickr8k/Annotations Flickr8k.token.txt
      ```
2. Generate files with image paths:
      ```
      python generate_img_lists.py --root-dir ~/DATASETS/Flickr8k/ --image-dir ~/DATASETS/Flickr8k/Images --input-suffix _list.txt --output-dir Annotations --output-suffix _list_images.txt --splits train val test
      ```

3. Process the images. You have two alternatives: using pre-extracted features or using the raw images.

    3.1 Alternative 1 - using feature vectors:

      3.1.1. Select the configuration of the extractor in `feature_extraction/config.py`. We'll extract features from the NASNetLarge model.

      3.1.2. Extract the features!

            
            python ../feature_extraction/keras/simple_extractor.py
            
            
      3.1.3. Generate lists pointing to the extracted features: `<split>_list_features.txt` (where <split> typically are train, val and test):
      
            
            python generate_feature_lists.py --root-dir DATASETS/Flickr8k --features-dir Features/Flickr8k_NASNetLarge/ --features NASNetLarge --lists-dir Annotations --extension .npy
            
        
      3.2. Alternative 2 - using raw images: modify the `config.py` accordingly.
