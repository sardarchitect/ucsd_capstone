import os
import cv2 as cv
import pandas as pd
import json
from utils.helper import dict_key_string_to_int

def get_directories():
    # Returns tuple of paths to the VIRAT dataset directories
    data_dir = '/mnt/d/data/virat/'
    annotation_dir = data_dir + 'annotations/'
    video_dir = data_dir + 'videos/'
    annotation_cols_dir = '/home/sardarchitect/repos/github.com/ucsd_capstone/streetstudy/data/anno/'
    return data_dir, annotation_dir, video_dir, annotation_cols_dir

def build():
    # Returns Pandas DataFrame containing VIRAT dataset video_names, paths, annotation files, and video duration
    _, annotation_dir, video_dir, _ = get_directories()

    if os.path.exists('./.data_cache/videos_df.pkl'):
        return pd.read_pickle('./.data_cache/videos_df.pkl')
    
    videos = [] 
    for _, video in enumerate(os.listdir(video_dir)):
        video_path = video_dir + video
        capture = cv.VideoCapture(video_path)
        fps = capture.get(cv.CAP_PROP_FPS)
        num_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        duration = int(num_frames/fps)
        videos.append([video.removesuffix('.mp4'), video_path, num_frames, duration])
        capture.release()

    videos_df = pd.DataFrame(videos, columns=['name', 'path', 'num_frames', 'duration'])
    videos_df.set_index('name', inplace=True)
    videos_df['event_file'] = ''
    videos_df['object_file'] = ''
    videos_df['mapping_file'] = ''
    for index, row in videos_df.iterrows():
        for file in os.listdir(annotation_dir):
            if row.name in file and "events" in file:
                videos_df.loc[index, 'event_file'] = file
            if row.name in file and "objects" in file:
                videos_df.loc[index, 'object_file'] = file
            if row.name in file and "mapping" in file:
                videos_df.loc[index, 'mapping_file'] = file

    videos_df.drop(videos_df[videos_df['event_file']==''].index, inplace=True)
    videos_df.drop(videos_df[videos_df['object_file']==''].index, inplace=True)
    videos_df.drop(videos_df[videos_df['mapping_file']==''].index, inplace=True)
    
    os.mkdir('./.data_cache')
    videos_df.to_pickle("./.data_cache/videos_df.pkl")
    return videos_df

def get_annotations(video_path, type='object'):
    # Returns a dataframe of the specified annotation file of the specified video from the VIRAT dataset

    _, annotation_dir, _, annotation_cols_dir = get_directories()
    # Load DF from cache, make cache folder if one does not exist
    if not os.path.exists('./.data_cache/videos_df.pkl'):
        os.mkdir('./.data_cache')
        videos_df = build()
        videos_df.to_pickle("./.data_cache/videos_df.pkl")
    else:
        videos_df = pd.read_pickle('./.data_cache/videos_df.pkl')
    
    # Find video in DF
    video = videos_df.loc[videos_df['path'] == video_path]
    
    with open(annotation_cols_dir+'objects.json') as json_file:
        objects = json.loads(json_file.read())

    with open(annotation_cols_dir+'objects_type.json') as json_file:
        objects_type = json.loads(json_file.read())
        objects = dict_key_string_to_int(objects)
    objects_type = dict_key_string_to_int(objects_type)

    annotation_df = pd.read_csv(annotation_dir + video['object_file'][0], delim_whitespace=True, header=None)
    annotation_df = annotation_df.rename(objects, axis=1)
    annotation_df = annotation_df[annotation_df['object_type'] == 1]

    return annotation_df