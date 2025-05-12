# Retinal Vessel Segmentation on DRIVE

**Author:** Mohammad Ahadzadeh  
**Date:** April 3, 2025  

A traditional image-processing pipeline to automatically segment blood vessels in retinal fundus photographs from the DRIVE dataset.

---

## Overview

This project implements a four-stage pipeline:

1. **Preprocessing**  
   - Extract green channel  
   - Contrast enhancement via CLAHE  
2. **Vessel Enhancement**  
   - Frangi filter for tubular structure detection  
3. **Segmentation**  
   - Otsu thresholding  
   - Morphological closing/opening  
   - Removal of small artifacts  
4. **Masking & Evaluation**  
   - Apply field-of-view mask  
   - Compute Accuracy, Sensitivity, Specificity, and Dice coefficient  

All steps are detailed in the Jupyter notebook; a standalone script batches the entire dataset and reports average metrics.
