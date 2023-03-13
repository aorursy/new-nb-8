# Need to add the library to Python path

import sys

sys.path.insert(0, "gif_your_nifti")
# Is it there?

# First, we need to move the .nii file to the output folder since it need to be writable 

# in addition to readable. 

# Not the cleanest import but it works :p

from gif_your_nifti.core import write_gif_pseudocolor

size = 1

fps = 20

cmap = 'hot'

write_gif_pseudocolor("fMRI_mask.nii", size, fps, cmap)
# Is the .gif here?

from IPython.display import Image

Image("fMRI_mask_hot.gif", width=720, height=480)