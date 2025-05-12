import os
import cv2
import numpy as np
from skimage import img_as_float
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import closing, disk, remove_small_objects
from PIL import Image


BASE_DIR = "DRIVE"

SUBDIRS = {
    'images': os.path.join(BASE_DIR, "training/images"),
    'mask': os.path.join(BASE_DIR, "training/mask"),
    'manual': os.path.join(BASE_DIR, "training/1st_manual"),
    'output': os.path.join(BASE_DIR, "training/segmented")
}
START_IDX, END_IDX = 21, 40

def load_image(path, mode='cv2'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if mode == 'pil':
        return np.array(Image.open(path))
    return cv2.imread(path)

def main():
    os.makedirs(SUBDIRS['output'], exist_ok=True)
    metrics = {'acc': [], 'sens': [], 'spec': [], 'dice': []}
    processed_files = 0

    for i in range(START_IDX, END_IDX + 1):
        # path
        files = {
            'image': f"{i:02d}_training.tif",
            'mask': f"{i:02d}_training_mask.gif",
            'manual': f"{i:02d}_manual1.gif"
        }

        try:
          
            img = load_image(os.path.join(SUBDIRS['images'], files['image']))
            mask = load_image(os.path.join(SUBDIRS['mask'], files['mask']), 'pil')
            gt = load_image(os.path.join(SUBDIRS['manual'], files['manual']), 'pil')

            # Preprocessing
            green_channel = img[:, :, 1]
            clahe = cv2.createCLAHE(clipLimit=5.5, tileGridSize=(32,32))
            enhanced = clahe.apply(green_channel)
            
            # Vessel enhancement
            vesselness = frangi(img_as_float(enhanced), sigmas=(1, 15), 
                              scale_step=2, beta=5, gamma=0.04)
            
            # Segmentation
            thresh = threshold_otsu(vesselness)
            binary = vesselness > thresh
            cleaned = closing(binary, disk(1))
            cleaned = remove_small_objects(cleaned, min_size=35)
            
            # Post-processing
            segmented = (cleaned * 255).astype(np.uint8)
            _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            final_seg = cv2.bitwise_and(segmented, segmented, mask=mask_binary)

            # Save
            output_path = os.path.join(SUBDIRS['output'], f"{i:02d}_segmented.png")
            cv2.imwrite(output_path, final_seg)

            # Result calculation
            seg_binary = final_seg > 0
            gt_binary = gt > 0
            
            tp = np.sum(seg_binary & gt_binary)
            fp = np.sum(seg_binary & ~gt_binary)
            fn = np.sum(~seg_binary & gt_binary)
            tn = np.sum(~seg_binary & ~gt_binary)

            # Avoiding zero division
            metrics['acc'].append((tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) else 0)
            metrics['sens'].append(tp / (tp + fn) if (tp + fn) else 0)
            metrics['spec'].append(tn / (tn + fp) if (tn + fp) else 0)
            metrics['dice'].append(2 * tp / (2 * tp + fp + fn) if (2*tp + fp + fn) else 0)
            
            processed_files += 1

        except Exception as e:
            print(f"Error processing image {i}: {str(e)}")
            continue

    # Results
    if processed_files > 0:
        print("Average Metrics:")
        print(f"Accuracy:    {np.nanmean(metrics['acc']):.4f}")
        print(f"Sensitivity: {np.nanmean(metrics['sens']):.4f}")
        print(f"Specificity: {np.nanmean(metrics['spec']):.4f}")
        print(f"Dice Coeff:  {np.nanmean(metrics['dice']):.4f}")
    else:
        print("No files processed successfully. Check input paths and files.")
if __name__ == "__main__":
    main()