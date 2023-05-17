'''
Prototype 1
Due: 05/21/2023

Event: Video uploaded by user
1. Storage(video)
2. OpenCV Video Capture
3. annotations = model(video)
4. postprocess(annotations)
5. Storage(heatmap.png)
6. Display heatmap
'''

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--video_path", type=str, required=True, help="Provide path of video to be processed")
    args = parser.parse_args()

    print('Processing:',args.video_path)
    # annotations = inference(args.video)
    # heatmap = postprocess(annotations)
    # print(heatmap)
if __name__ == '__main__':
    main()