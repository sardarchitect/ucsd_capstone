import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

directory = 'dataset/RPLAN/floorplan_dataset'

img = os.listdir(directory)[0]
img = os.path.join(directory, img)

print(len(os.listdir(directory)))

img = Image.open(img)
np_img = np.asarray(img)
print(np_img.shape)

# plt.imshow(np_img[:,:,:])
# plt.savefig(f'channel_all.png')

# for i in range(0,4):
#     plt.imshow(np_img[:,:,i])
#     plt.savefig(f'channel_{i}.png')

labels_xlsx = 'dataset/RPLAN/label.xlsx'
df = pd.read_excel(labels_xlsx)
print(df)
