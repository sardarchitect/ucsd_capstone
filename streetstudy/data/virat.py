import os
import cv2 as cv
import pandas as pd
import json
import numpy as np
from tqdm.auto import tqdm

def get_directories():
    # Returns tuple of paths to the VIRAT dataset directories
    data_dir = '/home/sardarchitect/repos/github.com/ucsd_capstone/virat/'
    annotation_dir = data_dir + 'annotations/'
    video_dir = data_dir + 'videos/'
    return data_dir, annotation_dir, video_dir

def get_annotation_col_names(name):
    # Returns column names for VIRAT dataset
    if name == "events_type":
        events_type = {
        1: "Person loading an Object to a Vehicle",
        2: "Person Unloading an Object from a Car/Vehicle",
        3: "Person Opening a Vehicle/Car Trunk",
        4: "Person Closing a Vehicle/Car Trunk",
        5: "Person getting into a Vehicle",
        6: "Person getting out of a Vehicle",
        7: "Person gesturing",
        8: "Person digging",
        9: "Person carrying an object",
        10: "Person running",
        11: "Person entering a facility",
        12: "Person exiting a facility"
        }
        return events_type
    elif name == "events":
        events = {
        0: "event_id",
        1: "event_type",
        2: "duration",
        3: "start_frame",
        4: "end_frame",
        5: "current_frame",
        6: "bbox_lefttop_x",
        7: "bbox_lefttop_y",
        8: "bbox_width",
        9: "bbox_height"
        }
        return events
    elif name == "mapping":
        mapping = {
        0: "event_id",
        1: "event_type",
        2: "event_duration",
        3: "start_frame",
        4: "end_frame",
        5: "num_obj"
        }
        return mapping
    elif name == "objects_type":
        objects_type = {
        0: "person",
        1: "car",
        2: "vehicles",
        3: "object",
        4: "bike, bicycles"
        }
        return objects_type
    elif name == "objects":
        objects = {
        0: "object_id",
        1: "object_duration",
        2: "current_frame",
        3: "bbox_lefttop_x",
        4: "bbox_lefttop_y",
        5: "bbox_width",
        6: "bbox_height",
        7: "object_type"
        }
        return objects
    else:
        print('Provide a valid column name')

def build_dataset():
    # Returns Pandas DataFrame containing VIRAT dataset video_names, paths, annotation files, and video duration
    _, annotation_dir, video_dir, _ = get_directories()

    # If DataFrame already exists, return DataFrame
    if os.path.exists('./.data_cache/videos_df.pkl'):
        return pd.read_pickle('./.data_cache/videos_df.pkl')
    
    # Create a temporary list to store video data
    videos_list = [] 
    for _, video in enumerate(os.listdir(video_dir)):
        video_name = video.removesuffix('.mp4')
        video_path = video_dir + video
        capture = cv.VideoCapture(video_path)
        fps = capture.get(cv.CAP_PROP_FPS)
        num_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        duration = int(num_frames/fps)
        
        for file in os.listdir(annotation_dir):
            if video_name in file and "events" in file:
                video_event_file = file
            if video_name in file and "objects" in file:
                video_object_file = file
            if video_name in file and "mapping" in file:
                video_mapping_file = file
        
        videos_list.append([video_name, video_path, num_frames, duration, video_event_file, video_object_file, video_mapping_file])
        capture.release()

    # Convert list to Pandas DataFrame and add additional information about annotation files
    videos_df = pd.DataFrame(videos_list, columns=['name', 'path', 'num_frames', 'duration', 'event_file', 'object_file', 'mapping_file'])
    videos_df.set_index('name', inplace=True)
    
    # Drop all videos that do not have annotations associated with them since they are not helpful for training/testing
    videos_df.drop(videos_df[videos_df['event_file']==''].index, inplace=True)
    videos_df.drop(videos_df[videos_df['object_file']==''].index, inplace=True)
    videos_df.drop(videos_df[videos_df['mapping_file']==''].index, inplace=True)
    
    # Store DataFrame in cache folder
    if not os.path.exists('./.data_cache'):
        os.mkdir('./.data_cache')
    videos_df.to_pickle("./.data_cache/videos_df.pkl")
    
    
    
    return videos_df

def get_annotations(video_path, type='object'):
    # Returns a dataframe of the specified annotation file of the specified video from the VIRAT dataset
    _, annotation_dir, = get_directories()
    videos_df = build_dataset()
    
    # Find video in DataFrame
    video = videos_df.loc[videos_df['path'] == video_path]
    # Find associated objects file
    annotation_df = pd.read_csv(annotation_dir + video['object_file'][0], delim_whitespace=True, header=None)
    # Rename columns
    objects_col = get_annotation_col_names('objects')
    annotation_df = annotation_df.rename(objects_col, axis=1)
    annotation_df = annotation_df[annotation_df['object_type'] == 1]
    return annotation_df

def virat_to_yolo(video, annotation_df, save_dir):
    # Converts VIRAT dataset to YOLO-specific folder structure
    if not os.path.exists(f'{save_dir}/labels'):
            os.mkdir(f'{save_dir}/labels')
    if not os.path.exists(f'{save_dir}/images'):
            os.mkdir(f'{save_dir}/images')
    capture = cv.VideoCapture(video.path)
    # Video size info
    image_width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    image_height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    df = annotation_df.copy()    
    
    # YOLO-standards
    df['object_type'] = 0
    df['bbox_center_x'] = df['bbox_lefttop_x'] + (df['bbox_width'] / 2)
    df['bbox_center_y'] = df['bbox_lefttop_y'] + (df['bbox_height'] / 2)   
    # Normalize to image size
    df['bbox_center_x'] = round(df['bbox_center_x'] / image_width, 3)
    df['bbox_center_y'] = round(df['bbox_center_y'] / image_height, 3)
    df['bbox_width'] = (df['bbox_width'] / image_width)
    df['bbox_height'] = (df['bbox_height'] / image_height)

    # Extract all frames that have annotations to jpg
    # Extract all annotations to txt
    for count in tqdm(range(video['num_frames'])):
        if count not in df['current_frame'].unique():
            continue    
        # Extract frame
        success, image = capture.read()
        if not success:
            print('Something went wrong')
            break
        cv.imwrite(f"{save_dir}/images/{video.name}_{count}.jpg", image)

        # Extract annotation
        bboxs = df[df['current_frame'] == count][['object_type', 'bbox_center_x', 'bbox_center_y', 'bbox_width', 'bbox_height']]
        np.savetxt(f'{save_dir}/labels/{video.name}_{count}.txt', bboxs.values, fmt='%.0f %.4f %.4f %.4f %.4f')