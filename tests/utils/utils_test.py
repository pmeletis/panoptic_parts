import numpy as np

import tifffile
from panoptic_parts.utils.utils import safe_write

pth0 = "test.png"
pth1 = "test1.png"
pth2 = "test2.png"
pth3 = "test3.png"
pth4 = "test4.png"

im = np.random.randint(0, high=255, size=(600, 800, 3), dtype=np.uint8)

# all following commands should have the same output file size
safe_write(pth0, im)
safe_write(pth1, im, optimize=True)
safe_write(pth2, im, compression_level=9)
tifffile.imwrite(pth3, im)
tifffile.imwrite(pth4, im, compression = 'zlib')
