import sys
from src import VideoAnalysis


def main(video_path, results_path):
    # Call function
    # /projects/p31496/Box
    VideoAnalysis(video_path, results_path)


if __name__ == '__main__':
    # quest_input contains files
    vid, res = sys.argv[1], sys.argv[2]
    main(vid, res)
