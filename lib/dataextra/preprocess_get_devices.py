# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import numpy as np
import argparse
import glob
import os.path as osp
from tqdm import tqdm
from lib.utils.utils import get_labeled_exif

# focal length in pixel from focal length in mm
#focal_pixel = (focal_length_mm / sensor_width_mm) * image_width_in_pixels

# focal length in pixel from horizontal field of view in degrees
#focal_pixel = (image_width_in_pixels * 0.5) / tan(FOV * 0.5 * math.pi/180)

def main(input_dir):

    paths = glob.glob(osp.join(input_dir, '**'), recursive=True)

    results = []
    for pth in tqdm(paths):
        make, model, lense_model, exif_height, exif_width, focal_length_mm, sensor_width = '','','','','', '', ''
        img_meta, focal_length_mm = get_labeled_exif(pth) 
        keys = img_meta.keys() 
        if 'Make' in keys: 
            make = img_meta['Make'] 
        if 'Model' in keys: 
            model = img_meta['Model'] 
        if 'LensModel' in keys:
            lense_model = img_meta['LensModel']
        if 'LenseInfo' in keys:
            print(img_meta['LenseInfo'])
        if 'CCD Width' in keys:
            sensor_width = keys['CCD Width']
        if 'ExifImageHeight' in keys: 
            exif_height, exif_width = img_meta['ExifImageHeight'], img_meta['ExifImageWidth'] 
        results += [make, model, lense_model, focal_length_mm, sensor_width, exif_width, exif_height] 

    res = np.array(results).reshape(-1,7)
    for row in res:
        print(row)

    devices_used = set([(x[0], x[1], x[2], x[3], x[4], x[5], x[6]) for x in res])
    for x in devices_used:
        print('{},{},{},{},{},{},{}'.format(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', required=True, help='folder with images to read EXIF from.')

    args = parser.parse_args()

    main(args.input_dir)
