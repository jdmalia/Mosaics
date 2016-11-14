# CS-6475 FINAL PROJECT
# Jason Malia (jmalia3)

import cv2
import numpy as np
import sys
import os

def read_images():
    
    library_folder = os.path.abspath(os.path.join(os.curdir, 'library'))
    main_folder = os.path.abspath(os.path.join(os.curdir, 'main_images'))
    
    # Extensions recognized by opencv
    exts = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg', 
            '.jpe', '.jp2', '.tiff', '.tif', '.png']

    # Read in main images
    print 'Searching for images in {} folder'.format(main_folder)
    filenames = os.listdir(os.path.join(main_folder))
    main_images = []
    
    for filename in filenames:
        name, ext = os.path.splitext(filename)
        if ext in exts:
            main_images.append(cv2.imread(os.path.join(main_folder, filename)))

    print 'Searching for images in {} folder'.format(library_folder)

    # For every image in the library directory
    filenames = os.listdir(os.path.join(library_folder))
    library_images = []

    for filename in filenames:
        name, ext = os.path.splitext(filename)
        if ext in exts:
            library_images.append(cv2.imread(os.path.join(library_folder, filename)))

    return main_images, library_images

def average_colors(image_list):
    colors = []
    for image in image_list:
        b, g, r = cv2.split(image)
        colors.append([np.mean(b), np.mean(g), np.mean(r)])
    
    return colors

def simple_mosaic(main_images, library_images):
    
    mosaics = []
    library_colors = average_colors(library_images)
    library_height = library_images[0].shape[0]
    library_width = library_images[0].shape[1]

    for source in main_images:
        print "Computing simple mosaic..."
        # Resize image so that library images will perfectly span it
        height = source.shape[0]
        width = source.shape[1]
        new_height = library_height * (height / library_height)
        new_width = library_width * (width / library_width)
        resized = cv2.resize(source,(new_width, new_height), interpolation = cv2.INTER_CUBIC)
        
        mosaic = np.zeros(resized.shape, dtype = np.float)
        # For each region of interest
        for i in range(0, new_height, library_height):
            for j in range(0, new_width, library_width):
                ROI = source[i:i+library_height, j:j+library_width]
                b, g, r = cv2.split(ROI)
                color = [np.mean(b), np.mean(g), np.mean(r)]
                min = np.finfo('float').max
                min_idx = 0
                for idx in range(len(library_colors)):
                    curr = 0
                    curr += (color[0]-library_colors[idx][0])**2
                    curr += (color[1]-library_colors[idx][1])**2
                    curr += (color[2]-library_colors[idx][2])**2
                    if curr < min:
                        min = curr
                        min_idx = idx
                mosaic[i:i+library_height, j:j+library_width] = library_images[min_idx]
    
        mosaics.append(mosaic)

    return mosaics

def random_mosaic(main_images, library_images, similarity_threshold):
    
    mosaics = []
    library_colors = average_colors(library_images)
    library_height = library_images[0].shape[0]
    library_width = library_images[0].shape[1]
    
    for source in main_images:
        print "Computing simple mosaic with randomness..."
        # Resize image so that library images will perfectly span it
        height = source.shape[0]
        width = source.shape[1]
        new_height = library_height * (height / library_height)
        new_width = library_width * (width / library_width)
        resized = cv2.resize(source,(new_width, new_height), interpolation = cv2.INTER_CUBIC)
        
        mosaic = np.zeros(resized.shape, dtype = np.float)
        # For each region of interest
        for i in range(0, new_height, library_height):
            for j in range(0, new_width, library_width):
                ROI = source[i:i+library_height, j:j+library_width]
                b, g, r = cv2.split(ROI)
                color = [np.mean(b), np.mean(g), np.mean(r)]
                min = np.finfo('float').max
                min_idx = 0
                idx = start_idx = np.random.randint(len(library_colors))
                found = False
                while (idx < len(library_colors) and not found):
                    curr = 0
                    curr += (color[0]-library_colors[idx][0])**2
                    curr += (color[1]-library_colors[idx][1])**2
                    curr += (color[2]-library_colors[idx][2])**2
                    if curr < min:
                        min = curr
                        min_idx = idx
                    if curr < similarity_threshold:
                        found = True
                        min_idx = idx
                    idx += 1
                idx = 0
                while (idx < start_idx and not found):
                    curr = 0
                    curr += (color[0]-library_colors[idx][0])**2
                    curr += (color[1]-library_colors[idx][1])**2
                    curr += (color[2]-library_colors[idx][2])**2
                    if curr < min:
                        min = curr
                        min_idx = idx
                    if curr < similarity_threshold:
                        found = True
                        min_idx = idx
                    idx += 1
                mosaic[i:i+library_height, j:j+library_width] = library_images[min_idx]
        
        mosaics.append(mosaic)

    return mosaics

def complex_mosaic(main_images, library_images):
    
    mosaics = []
    library_height = library_images[0].shape[0]
    library_width = library_images[0].shape[1]
    
    for source in main_images:
        print "Computing complex mosaic..."
        # Resize image so that library images will perfectly span it
        height = source.shape[0]
        width = source.shape[1]
        new_height = library_height * (height / library_height)
        new_width = library_width * (width / library_width)
        resized = cv2.resize(source,(new_width, new_height), interpolation = cv2.INTER_CUBIC)
        
        mosaic = np.zeros(resized.shape, dtype = np.float)
        # For each region of interest
        for i in range(0, new_height, library_height):
            for j in range(0, new_width, library_width):
                ROI = source[i:i+library_height, j:j+library_width]
                ROI = ROI.astype(np.float)
                min = np.finfo('float').max
                min_idx = 0
                for idx in range(len(library_images)):
                    curr = np.sum((ROI - library_images[idx])**2)**0.5
                    if curr < min:
                        min = curr
                        min_idx = idx
                mosaic[i:i+library_height, j:j+library_width] = library_images[min_idx]
        
        mosaics.append(mosaic)

    return mosaics

def rc_mosaic(main_images, library_images, similarity_threshold):
    
    mosaics = []
    library_height = library_images[0].shape[0]
    library_width = library_images[0].shape[1]
    
    for source in main_images:
        print "Computing complex mosaic with randomness..."
        # Resize image so that library images will perfectly span it
        height = source.shape[0]
        width = source.shape[1]
        new_height = library_height * (height / library_height)
        new_width = library_width * (width / library_width)
        resized = cv2.resize(source,(new_width, new_height), interpolation = cv2.INTER_CUBIC)
        
        mosaic = np.zeros(resized.shape, dtype = np.float)
        # For each region of interest
        for i in range(0, new_height, library_height):
            for j in range(0, new_width, library_width):
                ROI = source[i:i+library_height, j:j+library_width]
                ROI = ROI.astype(np.float)
                min = np.finfo('float').max
                min_idx = 0
                idx = start_idx = np.random.randint(len(library_images))
                found = False
                while (idx < len(library_images) and not found):
                    curr = np.sum((ROI - library_images[idx])**2)**0.5
                    if curr < min:
                        min = curr
                        min_idx = idx
                    if curr < similarity_threshold:
                        found = True
                        min_idx = idx
                    idx += 1
                idx = 0
                while (idx < start_idx and not found):
                    curr = np.sum((ROI - library_images[idx])**2)**0.5
                    if curr < min:
                        min = curr
                        min_idx = idx
                    if curr < similarity_threshold:
                        found = True
                        min_idx = idx
                    idx += 1
                mosaic[i:i+library_height, j:j+library_width] = library_images[min_idx]
        
        mosaics.append(mosaic)
    
    return mosaics

main_images, library_images = read_images()
simple_mosaics = simple_mosaic(main_images, library_images)
random_mosaics = random_mosaic(main_images, library_images, 400)
complex_mosaics = complex_mosaic(main_images, library_images)
rc_mosaics = rc_mosaic(main_images, library_images, 500)

outfolder = os.path.abspath(os.path.join(os.curdir, 'output'))
# Ensure that the directory that holds our output directories exists...
if not os.path.exists(outfolder):
    os.mkdir(outfolder)


print "writing output to {}".format(os.path.join(outfolder))
if not os.path.exists(os.path.join(outfolder, 'mosaics')):
    os.mkdir(os.path.join(outfolder, 'mosaics'))

for idx, image in enumerate(simple_mosaics):
    cv2.imwrite(os.path.join(outfolder,'mosaics','simple_mosaic{0:04d}.png'.format(idx)), image)

for idx, image in enumerate(random_mosaics):
    cv2.imwrite(os.path.join(outfolder,'mosaics','random_mosaic{0:04d}.png'.format(idx)), image)

for idx, image in enumerate(complex_mosaics):
    cv2.imwrite(os.path.join(outfolder,'mosaics','complex_mosaic{0:04d}.png'.format(idx)), image)
for idx, image in enumerate(rc_mosaics):
    cv2.imwrite(os.path.join(outfolder,'mosaics','rc_mosaic{0:04d}.png'.format(idx)), image)
