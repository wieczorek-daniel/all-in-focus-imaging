# Python All-in-Focus Imaging
## Introduction
Application of All-in-Focus image processing created by Python (OpenCV Python, NumPy and Tkinter).

## Pipeline
1) Choose directory with .png images for All-in-Focus image processing
2) Load image and the next image
3) Check if images are RGB or grayscale
4) Calculate gradient in x and y directions for each image
5) Calculate gradient for each pixel of both images
6) Calculate difference between image gradients for each pixel
7) Create filter matrix 3x3 filled with ones
8) Convolution filter with filter matrix and array from 6)
9) Decide pixel from which image is better and save result

If there is more than two files, image in the next iteration will be result of fused images from previous iteration. At the end save output .png file with fused images.  

## Setup
First install the dependencies (required packages) using following command:
```shell
pip3 install -r requirements.txt
```
Then run main .py script using command:
```shell
python all_in_focus.py
```

## Example result
### Input data:
| 1st image | 2nd image | 3rd image | 4th image |
|:---:|:---:|:---:|:---:|
|![1st image](data/grayscale_data/image_1.png)|![2nd image](data/grayscale_data/image_2.png)|![3rd image](data/grayscale_data/image_3.png)|![4th image](data/grayscale_data/image_1.png)|

### Output data:
<p align="center">
  <img src="result/grayscale_result/out_image.png">
</p>

## References
| No. | Reference | Source |
|:---:|:---:|:---:|
| 1. | "A Computationally Efficient Algorithm for Multi-Focus Image Reconstruction" </br> Authors: Helmy A. Eltoukhy and Sam Kavusi | [URL](https://www.researchgate.net/publication/228797652_A_computationally_efficient_algorithm_for_multi-focus_image_reconstruction) |
| 2. | Grayscale images data </br> Author: Samet Aymaz | [URL](https://github.com/sametaymaz/Multi-focus-Image-Fusion-Dataset) |
| 3. | RGB images data </br> Authors from University of Washington  Graphics and Imaging Laboratory | [URL](http://grail.cs.washington.edu/projects/photomontage/) |
