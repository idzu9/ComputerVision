## Homework 10

In this homework, you are going to use and compare two different trackers (of your liking) and compare the results.

### Step 1
Decide what video you are going to use for this homework, select an object and generate the template. You can use any video you want (your own, from Youtube, etc.)
and track any object you want (e.g. a car, a pedestrian, etc.).

### Step 2
Initialize a tracker (e.g. KCF).

### Step 3
Run the tracker on the video and the selected object. Run the tracker for around 10-15 frames.

### Step 4
For each frame, print the bounding box on the image and save it.

### Step 5
Select a different tracker (e.g. CSRT) and repeat steps 2, 3 and 4.

### Step 6
Compare the results:
* Do you see any differences? If so, what are they?
* Does one tracker perform better than the other? In what way?

I can say that I have performed the tests on those "KCF" and "CSRT" trackers by processing the whole video of the pedestrians_1.mp4 and I have the following observations:
* KCF and CSRT both start to track the selected region in the similar way
* But when the object is lost on the frames, sice there is a person covering the camare for some period of time and KCF tracker loses the object and never finds it again
* On the other hand CSRT still finds the object and continues to track it till the end of the video.
* So basically CSRT find the object on the image if it was lost for several frames, KCF does not track object once it was lost.

Some notes on how to use the homework

1. We load the videos from the data folder
2. Pick ROI of the object to track for each video by selecting it on the first frame of the video
3. then we run tests for the trackers and store results in the folder.

