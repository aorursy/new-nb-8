# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 




import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dicom

import os 

import scipy.ndimage

import matplotlib.pyplot as plt



from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection



INPUT_FOLDER = '../input/sample_images/'

patients = os.listdir(INPUT_FOLDER)

patients.sort()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Load scans

def load_scan(path):

    slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

       

    for s in slices:

        s.SliceThickness = slice_thickness

       

    return slices
def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])

    #Convert to int16

    #Should be possible as values should always be >32000

    image = image.astype(np.int16)

    

    #Set exterior domain pixels to 0

    # The intercept is usually -1024, so air is approx. 0

    image[image==-2000] =0

    

    #Convert to Hounsfield units (HU)

    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept

        slope = slices[slice_number].RescaleSlope

        

        if slope != 1:

            image[slice_number] = slope*image[slice_number].astype(np.float64)

            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

        

    return np.array(image, dtype=np.int16)
first_patient = load_scan(INPUT_FOLDER+patients[0])

first_patient_pixels = get_pixels_hu(first_patient)

plt.hist(first_patient_pixels.flatten(), bins=80, color='c')

plt.xlabel('Hounsfield Units (HU)')

plt.ylabel("Frequency")

plt.show()



#Show some slice in the middle

plt.imshow(first_patient_pixels[80],cmap=plt.cm.gray)

plt.show()
def resample(image, scan, new_spacing=[1,1,1]):

    #Determine current pixel spacing

    spacing = map(float, ([scan[0].SliceThickness]+ scan[0].PixelSpacing))

    spacing = np.array(list(spacing))

    

    resize_factor = spacing/new_spacing

    new_real_shape = image.shape*resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape/ image.shape

    new_spacing = spacing/real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image,real_resize_factor,mode='nearest')

    

    return image, new_spacing
pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])

print('Shape before resampling\t', first_patient_pixels.shape)

print('Shape after resampling\t', pix_resampled.shape)
def plot_3d(image, threshold=-300):

    

    # Position the scan upright,

    # so the head of the patient would be at the top, facing the camera

    p= image.transpose(2,1,0)

    

    verts,faces = measure.marching_cubes(p,threshold)

    

    fig = plt.figure(figsize=(10,10))

    ax=fig.add_subplot(111,projection='3d')

    

    # Fancy indexing: 'verts[faces] to generate a collection of triangles

    mesh = Poly3DCollection(verts[faces],alpha=0.1)

    face_color = (0.5,0.5,1)

    mesh.set_facecolor(face_color)

    ax.add_collection3d(mesh)

    

    ax.set_xlim(0,p.shape[0])

    ax.set_ylim(0,p.shape[1])

    ax.set_zlim(0,p.shape[2])

    

    plt.show()
plot_3d(pix_resampled, 400)
#Lung Segmentation portion

def largest_label_volume(im,bg=-1):

    vals, counts = np.unique(im,return_counts=True)

    

    counts = counts[vals != bg]

    vals = vals[vals != bg]

    

    if len(counts)>0:

        return vals[np.argmax(counts)]

    else:

        return None



    

def segment_lung_mask(image, fill_lung_structures = True):

    

    #Binary image: 1 or 2, 0 is background

    binary_image = np.array(image>-320, dtype = np.int8)+1

    labels = measure.label(binary_image)

    

    #Pick the pixel in the corner as air.

    #  Improvement: pick multiple background labels from around patient

    background_label = labels[0,0,0]

    

    binary_image[background_label == labels] = 2

    

    #Filling the the lung structures

    if fill_lung_structures:

        #For each slice, determine largest solid structure

        for i, axial_slice in enumerate(binary_image):

            axial_slice = axial_slice-1

            labeling = measure.label(axial_slice)

            l_max = largest_label_volume(labeling, bg=0)

            

            if l_max is not None:

                binary_image[i][labeling != l_max]=1

    binary_image -= 1 #make an actual binary image

    binary_image = 1-binary_image #Invert it so that lungs are now 1

    

    #Remove other air pockets

    labels = measure.label(binary_image, background=0)

    l_max = largest_label_volume(labels, bg=0)

    if l_max is not None: #There are air pockets

        binary_image[labels != l_max] =0

    

    return binary_image

    

    
segmented_lungs = segment_lung_mask(pix_resampled, False)

segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
plot_3d
def largest_label_volume(im, bg=-1):

    vals, counts = np.unique(im, return_counts=True)



    counts = counts[vals != bg]

    vals = vals[vals != bg]



    if len(counts) > 0:

        return vals[np.argmax(counts)]

    else:

        return None



def segment_lung_mask(image, fill_lung_structures=True):

    

    # not actually binary, but 1 and 2. 

    # 0 is treated as background, which we do not want

    binary_image = np.array(image > -320, dtype=np.int8)+1

    labels = measure.label(binary_image)

    

    # Pick the pixel in the very corner to determine which label is air.

    #   Improvement: Pick multiple background labels from around the patient

    #   More resistant to "trays" on which the patient lays cutting the air 

    #   around the person in half

    background_label = labels[0,0,0]

    

    #Fill the air around the person

    binary_image[background_label == labels] = 2

    

    

    # Method of filling the lung structures (that is superior to something like 

    # morphological closing)

    if fill_lung_structures:

        # For every slice we determine the largest solid structure

        for i, axial_slice in enumerate(binary_image):

            axial_slice = axial_slice - 1

            labeling = measure.label(axial_slice)

            l_max = largest_label_volume(labeling, bg=0)

            

            if l_max is not None: #This slice contains some lung

                binary_image[i][labeling != l_max] = 1



    

    binary_image -= 1 #Make the image actual binary

    binary_image = 1-binary_image # Invert it, lungs are now 1

    

    # Remove other air pockets insided body

    labels = measure.label(binary_image, background=0)

    l_max = largest_label_volume(labels, bg=0)

    if l_max is not None: # There are air pockets

        binary_image[labels != l_max] = 0

 

    return binary_image
segmented_lungs = segment_lung_mask(pix_resampled, False)

segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
plot_3d(segmented_lungs, 0)
plot_3d(segmented_lungs_fill, 0)
plot_3d(segmented_lungs_fill-segmented_lungs, 0)