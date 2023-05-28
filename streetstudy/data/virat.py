import os
import cv2 as cv
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

def get_dataset_directories():
    """
    Get absolute paths to the VIRAT dataset directories

    Keyword arguments:
    None
    
    Return:
    data_dir -- Parent data folder
    annotation_dir -- Annotations folder
    video_dir -- Raw videos folder
    """
    data_dir = '/home/sardarchitect/repos/github.com/ucsd_capstone/virat/'
    annotation_dir = data_dir + 'annotations/'
    video_dir = data_dir + 'videos/'
    return data_dir, annotation_dir, video_dir

def get_column_names(name):
    """
    Get column names for VIRAT dataset annotations

    Keyword arguments:
    name -- Type of annotation file

    Return:
    dict -- Dictionary of column names
    """
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

def get_dataset_df():
    """
    Get a Pandas DataFrame containtaing VIRAT dataset metadata
    
    Keyword Arguments:
    None

    Return:
    videos_df -- Pandas DataFrame containing VIRAT dataset video_names, paths, annotation files, and video duration'
    Creates a cache directory to store DataFrame in "pickle" format
    Returns DataFrame from cache if cache directory found
    """
    _, annotation_dir, video_dir = get_dataset_directories()

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
        image_width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
        image_height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)

        for file in os.listdir(annotation_dir):
            if video_name in file and "events" in file:
                video_event_file = file
            if video_name in file and "objects" in file:
                video_object_file = file
            if video_name in file and "mapping" in file:
                video_mapping_file = file
        
        videos_list.append([video_name, video_path, num_frames, duration, image_width, image_height, video_event_file, video_object_file, video_mapping_file])
        capture.release()

    # Convert list to Pandas DataFrame and add additional information about annotation files
    videos_df = pd.DataFrame(videos_list, columns=['name', 'path', 'num_frames', 'duration', 'image_width', 'image_height', 'event_file', 'object_file', 'mapping_file'])
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

def get_annotations_df(video_path, type='object', format='virat', normalize=False, object_id=False):
    """
    Returns annotations for a specific video in the VIRAT dataset in a given format

    Keyword Arguments:
    video_path -- path to video
    type -- annotation type to return (default "object")
    format -- format of DataFrame to return ("virat", "yolo") (default "virat")
    normalize -- returns coordinates normalized to image frame dimensions (default=False)
    object_id -- returns object_id

    Return:
    annotation_df -- A Pandas DataFrame for the specified annotation file for the specified video from the VIRAT dataset
    """
    _, annotation_dir, _ = get_dataset_directories()
    videos_df = get_dataset_df()
    
    # Find video in DataFrame
    video = videos_df.loc[videos_df['path'] == video_path]
    
    if type=='object':
        # Find associated objects file
        annotation_df = pd.read_csv(annotation_dir + video['object_file'][0], delim_whitespace=True, header=None)
        # Rename columns
        objects_col = get_column_names('objects')
        annotation_df = annotation_df.rename(objects_col, axis=1)
        annotation_df = annotation_df[annotation_df['object_type'] == 1]
    
    if format == 'yolo':
        return convert_virat_to_yolo(video, annotation_df, normalize=normalize, object_id=object_id)
    
    return annotation_df

def convert_virat_to_yolo(video, annotation_df, normalize, object_id=False):
    """
    Converts default VIRAT DataFrame format to standard YOLO format

    Keyword Arguments:
    video -- DataFrame containing standardized information of selected video
    annotation_df -- DataFrame containing annotations in standard VIRAT format
    normalize -- returns coordinates normalized to image frame dimensions (default=False) 
    object_id -- returns object_id
    
    Return:
    Pandas DataFrame containing VIRAT dataset annotations for a specified video in YOLO format
    """
    annotation_df = annotation_df.copy()
    annotation_df['object_type'] = 0

    annotation_df['bbox_center_x'] = annotation_df['bbox_lefttop_x'] + (annotation_df['bbox_width'] / 2)
    annotation_df['bbox_center_y'] = annotation_df['bbox_lefttop_y'] + (annotation_df['bbox_height'] / 2) 

    if normalize:
        # Normalize to image size
        annotation_df['bbox_center_x'] = round(annotation_df['bbox_center_x'] / video['image_width'][0], 3)
        annotation_df['bbox_center_y'] = round(annotation_df['bbox_center_y'] / video['image_height'][0], 3)
        annotation_df['bbox_width'] = (annotation_df['bbox_width'] / video['image_width'][0])
        annotation_df['bbox_height'] = (annotation_df['bbox_height'] / video['image_height'][0])

    if object_id:
        annotation_df.drop(['object_duration', 'bbox_lefttop_x', 'bbox_lefttop_y'], axis=1, inplace=True)
        annotation_df = annotation_df[['current_frame', 'object_id', 'object_type', 'bbox_center_x', 'bbox_center_y', 'bbox_width', 'bbox_height']]
    else:
        annotation_df.drop(['object_id', 'object_duration', 'bbox_lefttop_x', 'bbox_lefttop_y'], axis=1, inplace=True)
        annotation_df = annotation_df[['current_frame', 'object_type', 'bbox_center_x', 'bbox_center_y', 'bbox_width', 'bbox_height']]
    return annotation_df

def build_virat_to_yolo_directory(video, annotation_df, save_dir):
    """
    Builds a standardized YOLO directory for the VIRAT dataset at a specified location

    Keyword Arguments:
    video -- DataFrame containing standardized information of selected video
    annotation_df -- Pandas DataFrame of annotations for specific video in YOLO format
    save_dir -- path to directory

    Return:
    None
    """
    if not os.path.exists(f'{save_dir}/labels'):
            os.mkdir(f'{save_dir}/labels')
    
    if not os.path.exists(f'{save_dir}/images'):
            os.mkdir(f'{save_dir}/images')

    capture = cv.VideoCapture(video.path)
    
    for count in tqdm(range(video['num_frames'])):
        # Extract frames if and only if they contain annotations to save space
        if count not in annotation_df['current_frame'].unique():
            continue
        success, image = capture.read()
        if not success:
            print('Something went wrong while extracting video frames')
            break
        cv.imwrite(f"{save_dir}/images/{video.name}_{count}.jpg", image)

        # Extract annotation
        bboxs = annotation_df[annotation_df['current_frame'] == count][['object_type', 'bbox_center_x', 'bbox_center_y', 'bbox_width', 'bbox_height']]
        np.savetxt(f'{save_dir}/labels/{video.name}_{count}.txt', bboxs.values, fmt='%.0f %.4f %.4f %.4f %.4f')