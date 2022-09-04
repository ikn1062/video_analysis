import os
import cv2
from .zoom_data import ZoomVideo
from .database import VideoDatabase


class VideoAnalysis:
    def __init__(self, directory, results_path, num_participants=None, participants=None, db_info=None, fps=1):
        """
        Analyzes Zoom Video and extracts data to db or csv

        :param directory: Directory in which video files are stored - can be multiple (str)
        :param results_path: Path to where results should be stored (str)
        :param num_participants: [OPTIONAL] Number of participants per video (int)
        :param participants: [OPTIONAL] List of participants name for video - allows for name matching (list)
        :param db_info: [OPTIONAL] Database information - db_name, host, user, pass (tuple)
        :param fps: [OPTIONAL] Frames per second for video analysis (float)
        """
        self.directory = directory
        if type(self.directory) == str:
            self.directory = [self.directory]

        self.num_participants = num_participants
        if not self.num_participants:
            self.num_participants = [None] * len(self.directory)
        elif type(self.num_participants) == int:
            self.num_participants = [self.num_participants]

        self.participants = participants
        if not self.participants:
            self.participants = [None] * len(self.directory)
        elif type(self.participants) == list and type(self.participants[0]) == str:
            self.participants = [self.participants]

        self.results_path = results_path
        self.fps = fps

        self.db = None
        if db_info:
            self.__make_database(db_info)

        self.__check_init_error()

    def analyse_videos(self):
        """
        Calls ZoomVideo class and analyzes zoom video

        - Speaker data for each video is stored as a csv in Results
        - Accuracy data for each video is stored as a txt file in Results
        - Information will be uploaded to a MySQL db if db_info information is provided
        :return: None
        """
        for i, vid_directory in enumerate(self.directory):
            for filename in os.listdir(vid_directory):
                video_path = f"{vid_directory}/{filename}"
                if not self.__check_file_name(video_path):
                    print(video_path)
                    continue
                video = ZoomVideo(video_path, self.results_path, self.num_participants[i], self.participants[i],
                                  self.db, self.fps)
                try:
                    video.analyze_video()
                except FileExistsError:
                    video.del_known_faces()
                    video.recognition()
                    video.process_data()
                    video.del_known_faces()
                    os.remove(video.cropped_video_str)
        return 0

    def __check_init_error(self):
        """
        Checks for errors with the initialization of the VideoAnalysis Class
        :return: None
        """
        if self.num_participants[0]:
            assert self.num_participants[0] and len(self.directory) == len(self.num_participants), \
                "Warning: Length of directories should match length of number of participants."
        if self.participants[0]:
            assert len(self.directory) == len(self.participants), \
                "Warning: Length of directories should match length of participant list."
        assert type(self.results_path) == str, "Result path should be of type str"
        assert type(self.fps) == int and self.fps > 0, "fps input is invalid"
        assert type(self.participants) == list, "Participants should be of type list"
        return 0

    @staticmethod
    def __check_file_name(filename):
        """
        Checks for errors reading the specific file - returns error if the file is not a video or if cv2 returns an error
        :param filename: path to file
        :return: Whether filename is valid for video analysis (bool)
        """
        try:
            cap = cv2.VideoCapture(filename)
        except cv2.error:
            print(f"Error opening path / filename {filename}")
            return False
        try:
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading file path / filename {filename}")
                return False
        except cv2.error:
            print(f"Error reading file path / filename {filename}")
            return False

        return True

    def __make_database(self, db_info):
        """
        Makes a MySQL database using database information within db info
        :param db_info: A tuple of information to connect to MySQL Server and create DB - db_name, host, user, pass (tuple)
        :return: None
        """
        db_name, host, user, passwd = db_info
        self.db = VideoDatabase(db_name, host, user, passwd)
