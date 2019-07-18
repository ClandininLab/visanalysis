import glob
from skimage.external import tifffile

# Parameters
stack_fn = '/Users/minseung/Desktop/stack2.tiff'
imgs_dir = '/Users/minseung/Desktop/test'


if not imgs_dir[-1] == '/':
    imgs_dir = imgs_dir + '/'
with tifffile.TiffWriter(stack_fn) as stack:
    for filename in sorted(glob.glob(imgs_dir + '*.tif')):
        stack.save(tifffile.imread(filename))
