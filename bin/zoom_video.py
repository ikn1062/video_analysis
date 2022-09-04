from src import VideoAnalysis
from KEYS import MYSQL_PASS


def main():
    """
    Runs the Video Analysis of Zoom Video to capture speaker data

    Run the file with "python -m bin.zoom_video"
    :return: None
    """

    # Inputs:
    file_path = "data/zoom_data"
    video_path = f"{file_path}/videos"
    results_path = f"{file_path}/results"
    num_part = 6
    participants = None
    # db_info = ("test2", "localhost", "root", MYSQL_PASS)
    db_info = None

    # Function
    vid_data = VideoAnalysis(video_path, results_path, num_participants=num_part, participants=participants, db_info=db_info)
    vid_data.analyse_videos()


if __name__ == "__main__":
    main()
