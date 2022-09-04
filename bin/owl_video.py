from src import OwlVideo


def main():
    """
    Runs the Video Analysis of Zoom Video to capture speaker data
    Call with DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH" python -m bin.owl_video
    
    :return: None
    """
    video = "data/owl-data/videos/SLU2_1-1_part1.mkv"
    owl_vid = OwlVideo(video)
    owl_vid.analyze_video()
    owl_vid.analyze_vid_data()


if __name__ == "__main__":
    main()
