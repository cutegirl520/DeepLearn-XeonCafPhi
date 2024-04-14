#!/usr/bin/env python
from mincepie import mapreducer, launcher
import gflags
import os
import cv2
from PIL import Image

# gflags
gflags.DEFINE_string('image_lib', 'opencv',
                     'OpenCV or PIL, case insensitive. The default value is the faster OpenCV.')
gflags.DEFINE_string('input_folder', '',
                     'The folder that contains all input images, organized in synsets.')
gflags.DEFINE_integer('output_side_length', 256,
                     'Expected side length of the output image.')
gflags.DEFINE_string('output_folder', '',
                     'The folder that we write output resized and cropped images to')
FLAGS = gflags.FLAGS

class OpenCVResizeCrop:
    def resize_and_crop_image(self, input_file, output_file, output_side_length = 256):
        '''Takes an image name, resize it and crop the center square
        '''
        img = cv2.imread(input_file)
        height, width, depth = img.shape
        new_height = output_side_length
        new_width = output_side_length
        if height > width:
            new_height = output_side_length * height / width
        else:
            new_width = output_side_length * width / height
        resized_img = cv2.resize(img, (new_width, new_height))
        height_offset = (new_height - output_side_length) / 2
        width_offset = (new_width - output_side_length) / 2
        cropped_img = resized_img[height_offset:height_offset + output_side_length,
                                  width_offset:width_offset + output_side_length]
        cv2.imwrite(output_file, cropped_img)

class PILResizeCrop:
## http://united-coders.com/christian-harms/image-resizing-tips-every-coder-should-know/
    def resize_and_crop_image(self, input_file, output_file, output_side_length = 256, fit = True):
        '''Downsample the image.
        '''
        img = Image.open(input_file)
        box = (output_side_length, output_side_length)
        #preresize image with factor 2, 4, 8 and fast algorithm
        factor = 1
        while img.size[0]/factor > 2*box[0] and img.size[1]*2/factor > 2*box[1]:
            factor *=2
        if factor > 1:
            img.thumbnail((img.size[0]/factor, img.size[1]/factor), Image.NEAREST)

        #calculate the cropping box and get the cropped part
        if fit:
            x1 = y1 = 0
           