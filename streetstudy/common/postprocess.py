import matplotlib.pyplot as plt
import matplotlib.patches
import seaborn as sns
import numpy as np

def heatmap(ax, preds, current_frame_number):
    preds = preds[preds[:, 0] <= current_frame_number]
    feet_x = preds[:, 1] + ((preds[:, 3] - preds[:, 1]) / 2)
    feet_y = preds[:, 2] + (preds[:, 4] - preds[:, 2])
    sns.kdeplot(x=feet_x, y=feet_y, thresh=0, levels=50, alpha=0.2, fill=False, ax=ax, cmap='hot')
    return ax

def bounding_boxes(ax, preds, current_frame_number):
    preds = preds[preds[:, 0] == current_frame_number]
    for bbox in preds:
        ax.text(bbox[1], bbox[2], bbox[5])
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (bbox[1], bbox[2]), 
                bbox[3] - bbox[1], 
                bbox[4] - bbox[2], 
                facecolor='none', 
                ec='r', 
                lw=1)
        )
    return ax

def directional_arrows(ax, preds, current_frame_number):
    preds = preds[preds[:, 0] <= current_frame_number]    
    unique_obj = np.unique(preds[:,5])
    cmap = get_cmap(len(unique_obj))
    for i in unique_obj:
        bbox = preds[preds[:, 5] == i]
        feet_x = bbox[:, 1] + ((bbox[:, 3] - bbox[:, 1]) / 2)
        feet_y = bbox[:, 2] + (bbox[:, 4] - bbox[:, 2])
        feet_x_next = np.delete(feet_x, 0, 0)
        feet_x_next = np.append(feet_x_next, feet_x[-1])
        feet_y_next = np.delete(feet_y, 0, 0)
        feet_y_next = np.append(feet_y_next, feet_y[-1])
        dX = (feet_x_next - feet_x)
        dY = (feet_y_next - feet_y)
        for i in range(len(feet_x)):
            ax.arrow(feet_x[i], feet_y[i], dX[i], dY[i], length_includes_head=True,  head_starts_at_zero=True, head_width=3, alpha=0.4, color=cmap(i))
    return ax

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)