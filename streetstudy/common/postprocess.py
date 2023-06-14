import matplotlib.pyplot as plt
import matplotlib.patches
import seaborn as sns
import numpy as np

def heatmap(preds, current_frame_number, ax=None):
    """
    Generate a heatmap plot based on the given predictions up to the specified frame number.

    Args:
        preds (numpy.ndarray): Array of predictions containing coordinates and other information.
        current_frame_number (int): Current frame number to consider for generating the heatmap.
        ax (matplotlib.axes.Axes, optional): Axes object to plot the heatmap on. If not provided, the current Axes object will be used.

    Returns:
        matplotlib.axes.Axes: Axes object containing the heatmap plot.
    """
    preds = preds[preds[:, 0] <= current_frame_number]
    feet_x = preds[:, 1] + ((preds[:, 3] - preds[:, 1]) / 2)
    feet_y = preds[:, 2] + (preds[:, 4] - preds[:, 2])
    
    if ax is None:
        ax = plt.gca()

    ax = sns.kdeplot(
        x=feet_x, 
        y=feet_y,
        thresh=0, 
        levels=50, 
        alpha=0.2, 
        fill=False, 
        ax=ax,
        cmap='hot'
    )
    
    return ax

def bounding_boxes(preds, current_frame_number, ax=None):
    """
    Generate bounding box annotations for the specified frame number based on the given predictions.

    Args:
        preds (numpy.ndarray): Array of predictions containing coordinates and other information.
        current_frame_number (int): Frame number to consider for generating the bounding box annotations.
        ax (matplotlib.axes.Axes, optional): Axes object to plot the bounding boxes on. If not provided, the current Axes object will be used.

    Returns:
        matplotlib.axes.Axes: Axes object containing the bounding box annotations.
    """
    preds = preds[preds[:, 0] == current_frame_number]

    if ax is None:
        ax = plt.gca()

    for bbox in preds:
        ax.text(bbox[1], bbox[2], bbox[5])
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (bbox[1], bbox[2]), 
                bbox[3] - bbox[1], 
                bbox[4] - bbox[2], 
                facecolor='none', 
                ec='r', 
                lw=1
            )
        )
    
    return ax

def directional_arrows(preds, current_frame_number, ax=None):
    """
    Generate directional arrows based on the given predictions up to the specified frame number.

    Args:
        preds (numpy.ndarray): Array of predictions containing coordinates and other information.
        current_frame_number (int): Current frame number to consider for generating the directional arrows.
        ax (matplotlib.axes.Axes, optional): Axes object to plot the directional arrows on. If not provided, the current Axes object will be used.

    Returns:
        matplotlib.axes.Axes: Axes object containing the directional arrows.
    """
    preds = preds[preds[:, 0] <= current_frame_number]    
    unique_obj = np.unique(preds[:,5])
    cmap = get_cmap(len(unique_obj))
    
    if ax is None:
        ax = plt.gca()

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
            ax.arrow(
                feet_x[i],
                feet_y[i],
                dX[i],
                dY[i],
                length_includes_head=True,
                head_starts_at_zero=True,
                head_width=3,
                alpha=0.4,
                color=cmap(i)
            )

    return ax

def get_cmap(n, name='hsv'):
    """
    Get a colormap based on the given number of colors and colormap name.

    Args:
        n (int): Number of colors for the colormap.
        name (str, optional): Name of the colormap. Defaults to 'hsv'.

    Returns:
        matplotlib.colors.Colormap: Colormap object.
    """
    return plt.cm.get_cmap(name, n)