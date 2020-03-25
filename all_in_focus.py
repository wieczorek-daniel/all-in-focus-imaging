# Imports
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import sys
import fnmatch
import os
import cv2
import numpy as np
import warnings


# Function for All-in-Focus RGB images (dim = 3)
def rgb_image():
    for dim in range(image_size[2]):
        # i represents row number, j represents col number, dim represents dimension number
        for i in range(image_size[0]):
            for j in range(image_size[1]):
                # Calculate gradient for each pixel of image
                grad[i, j, dim] = abs(grad_x[i, j, dim]) + abs(grad_y[i, j, dim])
                # Calculate gradient for each pixel of next image
                next_grad[i, j, dim] = abs(next_grad_x[i, j, dim]) + abs(next_grad_y[i, j, dim])
                # Calculate difference between two gradients for each pixel
                matrix[i, j, dim] = grad[i, j, dim] - next_grad[i, j, dim]

        # Filter matrix 3x3 filled with ones
        filter_matrix = np.ones((3, 3))
        # Convolution filter
        image_filter = cv2.filter2D(matrix, -1, filter_matrix, borderType=cv2.BORDER_CONSTANT)
        # β parameter - the best results for β about 0.1
        beta = 0.1

        for i in range(image_size[0]):
            for j in range(image_size[1]):
                # Calculate sigmoid filter function for each pixel (approximately 0 or 1 values)
                image_filter[i, j, dim] = (1 / (1 + np.exp(-beta * image_filter[i, j, dim])))
                ''' Choose better pixel - if image_filter[i, j] ≈ 0 take pixel from image, 
                    else if image_filter[i, j] ≈ 1 take pixel from next image '''
                result[i, j, dim] = image_filter[i, j, dim] * image[i, j, dim] + (1 - image_filter[i, j, dim]) * next_image[i, j, dim]
    return result


# Function for All-in-Focus grayscale images (dim = 1)
def grayscale_image():
    # i represents row number, j represents col number
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            # Calculate gradient for each pixel of image
            grad[i, j] = abs(grad_x[i, j]) + abs(grad_y[i, j])
            # Calculate gradient for each pixel of next image
            next_grad[i, j] = abs(next_grad_x[i, j]) + abs(next_grad_y[i, j])
            # Calculate difference between two gradients for each pixel
            matrix[i, j] = grad[i, j] - next_grad[i, j]

    # Filter matrix 3x3 filled with ones
    filter_matrix = np.ones((3, 3))
    # Convolution filter
    image_filter = cv2.filter2D(matrix, -1, filter_matrix, borderType=cv2.BORDER_CONSTANT)
    # β parameter - the best results for β about 0.1
    beta = 0.1

    for i in range(image_size[0]):
        for j in range(image_size[1]):
            # Calculate sigmoid filter function for each pixel (approximately 0 or 1 values)
            image_filter[i, j] = (1 / (1 + np.exp(-beta * image_filter[i, j])))
            ''' Choose better pixel - if image_filter[i, j] ≈ 0 take pixel from image, 
                else if image_filter[i, j] ≈ 1 take pixel from next image '''
            result[i, j] = image_filter[i, j] * image[i, j] + (1 - image_filter[i, j]) * next_image[i, j]
    return result


# Choose path of directory with images
root = tk.Tk()
root.withdraw()
path = filedialog.askdirectory()

# Check if directory was not chosen
if path == '':
    # Show error message
    messagebox.showerror(title='Error', message='Directory was not chosen.')
    sys.exit(1)

# Calculate number of .png files in chosen directory
files_number = len(fnmatch.filter(os.listdir(path), '*.png'))

# Check if directory is empty
if files_number == 0:
    # Show error message
    messagebox.showerror(title='Error', message='Directory is empty (application supports only .png images)')
    sys.exit(1)

# Add \ (for Windows) or / (for Unix) to directory path
if os.name in ('nt', 'dos'):
    path += '\\'
else:
    path += '/'

# Define number of processing file
file_number = 0
# Ignore RuntimeWarning for overflow encountered in exp
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Loop for all images in chosen directory
for file_name in os.listdir(path):
    # Check if file is .png image
    if file_name.endswith('.png'):
        file_number += 1
        # Show progress message
        message = 'Image processing: image '+str(file_number)+' of '+str(files_number)
        print(message)

        # Image in next iteration is result of fused images from previous iteration
        if file_number != 1:
            image = result
        # Load first image
        else:
            image = cv2.imread(path + file_name,  cv2.CV_64F)
            image_size = image.shape
            result = np.zeros(image_size)
            continue

        # Initialize result array with zeros
        result = np.zeros(image_size)
        # Load next image
        next_image = cv2.imread(path + file_name,  cv2.CV_64F)

        # Calculate gradient in the x direction for image
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        # Calculate gradient in the y direction for image
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        # Calculate gradient in the x direction for next image
        next_grad_x = cv2.Sobel(next_image, cv2.CV_64F, 1, 0, ksize=5)
        # Calculate gradient in the y direction for next image
        next_grad_y = cv2.Sobel(next_image, cv2.CV_64F, 0, 1, ksize=5)

        # Initialize arrays with zeros
        grad = np.zeros(image_size)
        next_grad = np.zeros(image_size)
        matrix = np.zeros(image_size)

        # Check if image RGB
        if len(image_size) == 3:
            result = rgb_image()
        else:
            result = grayscale_image()

# Save output file with fused images
out_file_name = 'out_image.png'
cv2.imwrite(out_file_name, result)
# Show success message
final_message = 'Image saved in directory: '+os.path.abspath(os.getcwd())+' as: '+out_file_name
messagebox.showinfo(title='Success', message=final_message)
