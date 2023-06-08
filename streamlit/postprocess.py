import matplotlib.pyplot as plt
import matplotlib.patches
import seaborn as sns
import streamlit as st
import plotly as plty
import plotly.express as plty_exp
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
        plt.text(bbox[1], bbox[2], bbox[5])
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (bbox[1], bbox[2]), 
                bbox[3] - bbox[1], 
                bbox[4] - bbox[2], 
                rotation_point='xy',
                facecolor='none', 
                ec='r', 
                lw=1)
        )
    return ax

def directional_arrows(ax, preds, current_frame_number):
    preds = preds[preds[:, 0] <= current_frame_number]    
    unique_obj = np.unique(preds[:,5])
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
            ax.arrow(feet_x[i], feet_y[i], dX[i], dY[i], length_includes_head=True,  head_starts_at_zero=True, head_width=1, alpha=0.2)
    return ax

def show_plot(pix):
    img_rgb = np.random.randint(low=0, high=255, size=(pix, pix))
    fig = plty_exp.imshow(img_rgb)
    fig.add_annotation(text=st.session_state["display_type"], showarrow=False, x=-1, y=-1)
    st.plotly_chart(fig)

def plot_dwell():
    x = np.arange(start=0, stop=100, step=0.5)
    y = np.sin(x)
    fig = plty_exp.line(x=x,y=y, height=300)
    st.plotly_chart(fig, use_container_width=True)