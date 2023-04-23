import argparse
import pandas as pd
from display.display import display_raw_video, display_annotated_video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", type=str, required=True, help="Provide path of video to be processed")
    parser.add_argument("-r", "--raw_video", action=argparse.BooleanOptionalAction, help="Display raw video")
    parser.add_argument("-vi", "--is_virat", action=argparse.BooleanOptionalAction, help="Display annotations from VIRAT dataset")
    args = parser.parse_args()

    # If raw video requested, display raw video
    if args.raw_video:
        display_raw_video(args.video_path)
    
    if args.is_virat:
        display_annotated_video(args.video_path, virat=True)

if __name__ == '__main__':
    main()