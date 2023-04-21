import cv2 as cv
import pandas as pd
from Archive.annotations_columns import *

def rescaleFrame(frame, scale=0.25):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def virat_txt_to_df(annotations_dir, current_video, table_category):
    if table_category == 'events':
        column_names = events_col_dict
    if table_category == 'objects':
        column_names = objects_col_dict
    if table_category == 'mapping':
        column_names = mapping_col_dict
    
    f = annotations_dir + f'{current_video}.viratdata.{table_category}.txt'
    df = pd.read_csv(f, delim_whitespace=True, header=None)
    df = df.rename(column_names, axis=1)
    return df

def get_bbox(df, current_frame):
    bbox_list = []
    # GET BBOX COORDS FOR DISPLAY
    for index, row in (df[df['current_frame'] == current_frame]).iterrows():
        bbox_lefttop_x = row["bbox_lefttop_x"]
        bbox_lefttop_y = row["bbox_lefttop_y"]
        bbox_width = row["bbox_width"]
        bbox_height = row["bbox_height"]
        bbox_rightbottom_x = bbox_lefttop_x + bbox_width
        bbox_rightbottom_y = bbox_lefttop_y + bbox_height
        bbox_list.append([bbox_lefttop_x, bbox_lefttop_y, bbox_rightbottom_x, bbox_rightbottom_y, row["object_type"]])
    return bbox_list


def get_dirs():
    data_dir = './data/VIRAT Ground Dataset/'
    annotations_dir = data_dir + 'annotations/'
    videos_dir = data_dir + 'videos_original/'
    return data_dir, annotations_dir, videos_dir