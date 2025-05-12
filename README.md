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

## Resources

- “DRIVE Digital Retinal Images for Vessel Extraction.” Accessed: Apr. 03, 2025. [Online]. Available: https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction
- [1] A. Khanal and R. Estrada, “Dynamic Deep Networks for Retinal Vessel Segmentation,” *Frontiers in Computer Science*, vol. 2, Aug. 2020. doi: 10.3389/fcomp.2020.00035.
- [2] N. Memari, A. R. Ramli, M. I. B. Saripan, S. Mashohor, and M. Moghbel, “Retinal Blood Vessel Segmentation by Using Matched Filtering and Fuzzy C-means Clustering with Integrated Level Set Method for Diabetic Retinopathy Assessment,” *Journal of Medical and Biological Engineering*, vol. 39, no. 5, pp. 713–731, Oct. 2019. doi: 10.1007/s40846-018-0454-2.
- [3] S. Mahapatra, S. Agrawal, P. K. Mishro, and R. B. Pachori, “A Novel Framework for Retinal Vessel Segmentation Using Optimal Improved Frangi Filter and Adaptive Weighted Spatial FCM,” *Computers in Biology and Medicine*, vol. 147, p. 105770, Aug. 2022. doi: 10.1016/j.compbiomed.2022.105770.
- [4] F. Tian, Y. Li, J. Wang, and W. Chen, “Blood Vessel Segmentation of Fundus Retinal Images Based on Improved Frangi and Mathematical Morphology,” *Computational and Mathematical Methods in Medicine*, vol. 2021, p. 4761517, May 2021. doi: 10.1155/2021/4761517.


All steps are detailed in the Jupyter notebook; a standalone script batches the entire dataset and reports average metrics.
