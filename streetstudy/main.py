import argparse
from utils.display import display_raw_video, display_annotated_video, get_annotations
from model import yolov5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--video_path", type=str, required=True, help="Provide path of video to be processed")
    parser.add_argument("-r", "--raw", action=argparse.BooleanOptionalAction, help="Display raw video")
    parser.add_argument("-vi", "--virat", action=argparse.BooleanOptionalAction, help="Display annotations from VIRAT dataset")
    parser.add_argument("-inf", "--inference", action=argparse.BooleanOptionalAction, help="Display annotations from trained model")        
    args = parser.parse_args()

    if args.raw:
        display_raw_video(args.video_path)

    if args.virat:
        annotations_df = get_annotations(args.video_path)
        display_annotated_video(args.video_path, annotations_df)

    if args.inference:
        yolov5.inference(args.video_path)

if __name__ == '__main__':
    main()