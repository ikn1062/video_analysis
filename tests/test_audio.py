from pyannote.audio import Pipeline
import subprocess


def main():
    """
    # call with env DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH" python -m src.owl_data.owl_audio
    """
    video_path = "data/owl-data/videos"
    video_name = "SLU2_1-1_part1.mkv"
    audio_name = "audio_out.wav"
    command = f"ffmpeg -i ./{video_path}/{video_name} -ab 160k -ac 2 -ar 44100 -vn ./{video_path}/{audio_name}"
    subprocess.call(command, shell=True)

    print("Getting Pipeline")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

    # apply pretrained pipeline
    print("Diarization")
    diarization = pipeline(f"{video_path}/{audio_name}")

    print("result")
    # print the result
    speaker_dir = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        speaker_dir.append([turn.start, turn.end, speaker])
    print(speaker_dir)


if __name__ == "__main__":
    main()
