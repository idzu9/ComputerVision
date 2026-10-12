#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

# constants
data_folder_path = "data/"
videos_names = ["pedestrians_1.mp4", "pedestrians_2.mp4"]
tracker_types = ["KCF", "CSRT"]
# regulate the number of frames to process, 0 means full video
number_of_frames_to_track = 15

# load the vide and its number of frames and fps
def load_video_file(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
       cap.release()
       print(f"Could not open path for: ", video_path)
       return []

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    video_frames = []

    while True:
        ret, frame = cap.read()
        # end of video
        if not ret:
            break;
        video_frames.append(frame)

    print(f"Caprured video with ", len(video_frames), " frames with ", fps, " fps")
    cap.release()

    return video_frames, len(video_frames), fps

# load the videos and store it in structure
def load_videos(path, video_names):
    videos = {}
    for video in video_names:
        frames, total_frames, fps = load_video_file(path + video)

        if frames:
            videos[video] = {
                "frames": frames,
                "total_frames": total_frames,
                "fps": fps
            }
    return videos;

# play the video file
def play_video(video):
    delay = round(1000/video["fps"])

    for frame in video["frames"]:
        cv2.imshow("Loaded video",  frame)
        if (cv2.waitKey(delay)) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    return

# select the ROI for the pedestrian to track to
def select_pedestrian_roi(video_to_detect_object):
    pedestrian_box = cv2.selectROI("Select Target", video_to_detect_object["frames"][0], fromCenter=False, showCrosshair=False)
    cv2.destroyWindow("Select Target")
    x, y, w, h = pedestrian_box
    selected_pedestrian = video_to_detect_object["frames"][0][y:y+h, x:x+w]

    cv2.imshow("Selected pedestrian", selected_pedestrian)
    if (cv2.waitKey(0)) & 0xFF == 27:
        cv2.destroyAllWindows()

    return pedestrian_box

def run_tracker_tests(videos, selected_pedestrians):
    # for each video we will run the test for each tracker
    for video_name, video in videos.items():

        for tracker_type in tracker_types:
            tracker = setup_tracker(tracker_type)
            tracker.init(video["frames"][0], selected_pedestrians[video_name])

            nb_frames_to_track = number_of_frames_to_track

            if nb_frames_to_track <= 0:
                nb_frames_to_track = len(video["frames"])

            for i in range(nb_frames_to_track):
                frame = video["frames"][i].copy()
                ok, bbox = tracker.update(frame)
                print(ok, bbox)
                if ok == True:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                output_dir = os.path.join(data_folder_path, tracker_type)
                os.makedirs(output_dir, exist_ok=True)
                screenshot_filename = f"{tracker_type}_{video_name}_frame_{i}.png"
                screenshot_path = os.path.join(output_dir, screenshot_filename)
                print(screenshot_path)
                cv2.imwrite(screenshot_path, frame)

def main(args):    
    # step 1 load the video
    videos = load_videos(data_folder_path, videos_names)

    # step 2 select the pedestrian on the first frame of the video
    selected_pedestrians = {}
    for video_name, video in videos.items():
        selected_pedestrians[video_name] = select_pedestrian_roi(video)

    # step 3 run tracker tests for selected ROI
    run_tracker_tests(videos, selected_pedestrians)

def setup_tracker(tracker_type):
    if tracker_type == "KCF":
        tracker = cv2.TrackerKCF_create()

    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    return tracker

if __name__ == '__main__':
    main(sys.argv)