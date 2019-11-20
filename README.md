Welcome to the comma.ai 2017 Programming Challenge!

Basically, your goal is to predict the speed of a car from a video.

data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
data/train.txt contains the speed of the car at each frame, one speed on each line.

data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.
Your deliverable is test.txt

We will evaluate your test.txt using mean squared error. <10 is good. <5 is better. <3 is heart.

---

`get_features.py` uses dense optical flow to determine the delta between any pair of consecutive frames, then
crops/resizes the delta image to get a feature vector. It does this for every pair of consecutive frames in the video
and writes all the feature vectors to `data/{}_frames.npy`.

`main.py` creates a convolutional neural net, reads in feature vectors and labels from `data/`, and trains the net on that data. 
It then validates the model and uses the model to predict speeds for `data/test_frames.npy`.

To run the model:
1. Make sure all dependencies are installed (opencv, keras, tf, etc.)
2. Put `train.mp4` and `test.mp4` in `data/` (not included in repo due to Github's file limit)
3. `python3 get_features.py train.mp4`
4. `python3 get_features.py test.mp4`
5. `python3 main.py`