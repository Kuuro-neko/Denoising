"""
Sequential execution of preprocessing scripts:
1. drop_flickr_images.py - Keep only first N images from Flickr dataset
2. patches_processor.py - Extract patches and apply different noise types
3. noise_dataset.py - Resize clean patches to 64x64 and create noisy versions
"""

import drop_flickr_images
import patches_processor
import noise_dataset
import os

def main():
    if not os.path.exists("patches"):
        os.makedirs("patches")
    if not os.path.exists("patchesx64"):
        os.makedirs("patchesx64")
    if not os.path.exists("flickr_images"):
        print("Flickr images must be placed in the 'flickr_images' directory before running this script.")
        return

    print("=" * 80)
    print("STEP 1: Filtering Flickr images")
    print("=" * 80)
    drop_flickr_images.keep_first_n_images("flickr_images", 5000)
    
    print("\n" + "=" * 80)
    print("STEP 2: Processing patches from Flickr images")
    print("=" * 80)
    patches_processor.main()
    
    print("\n" + "=" * 80)
    print("STEP 3: Creating 64x64 dataset with random noise")
    print("=" * 80)
    noise_dataset.main()
    
    print("\n" + "=" * 80)
    print("ALL PREPROCESSING STEPS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
