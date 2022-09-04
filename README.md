# Video Speaker Analysis

## Introduction

This library is created to analyze Zoom and Owl Labs Video data to recognize who is speaking
at a given point in time. The reason for creating this library was to aid researches in collecting
data from Zoom and Owl video data (virtual and in-person videos) to analyze speaker interactions
and how it leads to future relationships.

## Installation
To install and use this package, do the following:

```angular2html
git clone https://github.com/ikn1062/dynamics-of-conferences-video-data.git
python setup.py install
```
or
```angular2html
git clone https://github.com/ikn1062/dynamics-of-conferences-video-data.git
pip install -r requirements.txt
```

## Run Scripts
To run a Video Analysis of a given zoom video, you can use the script in `bin/zoom_video`

To run this script, edit the `video_path`, `results_path`, `participants`, `num_participants`,
`db_info` fields depending on your analysis configuration, and run:
```angular2html
python -m bin.zoom_video
```

## Zoom Video Analysis
### Introduction
The Zoom Video Analysis uses the `face_recognition` and `dlib` library, as well as the 
`pytesseract` optimal character recognition to successfully recognize the name of the 
speaker and the face of the speaker for each frame. Combining both recognition methods allows
for a 99%+ accuracy in speaker recognition for Zoom Videos.

### VideoAnalysis Class
Analyzes Zoom Video and extracts data to db or csv
```
:param directory: Directory in which video files are stored - can be multiple (str)
:param results_path: Path to where results should be stored (str)
:param num_participants: [OPTIONAL] Number of participants per video (int)
:param participants: [OPTIONAL] List of participants name for video - allows for name matching (list)
:param db_info: [OPTIONAL] Database information - db_name, host, user, pass (tuple)
:param fps: [OPTIONAL] Frames per second for video analysis (float)
```


## Owl Video Analysis
The Owl Video Analysis uses `dlib` `face_recognition`, `mediapipe`, and `pyannote.audio` libraries
to recognize the speaker at any given frame. The `pyannote.audio` speaker dirization tool allows 
for a speaker-based recognition which can be linked to a given speaker using the `mediapipe`
face mesh to track mouth movements and the `face_recognition` library.
