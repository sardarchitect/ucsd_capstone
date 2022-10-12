import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

directory = 'dataset/RPLAN/floorplan_dataset'

img = os.listdir(directory)[0]
img = os.path.join(directory, img)

img = Image.open(img)
np_img = np.asarray(img)
print(np_img.shape)

plt.imshow(np_img[:,:,1])
plt.savefig('output.png')
plt.show()