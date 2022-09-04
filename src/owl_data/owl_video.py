import cv2
import os
import numpy as np
import face_recognition as fr
from pyannote.audio import Pipeline
import pickle
import shutil
import mediapipe as mp
import subprocess
import pyboof as pb
from collections import defaultdict
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class OwlVideo:
    def __init__(self, vid_str, fps=5):
        """
        Analyzes Owl Video to extract data on who is speaking at what time
        - Cleans video data and extracts video frame data to csv and accuracy of analysis to txt in results folder

        Call with DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH" python -m src.owl_data.owl_video

        :param vid_str: Path to video file (str)
        :param fps: [OPTIONAL] FPS to analyze video with - set to 5 fps (float)
        """
        # Get file path and start video capture
        self.vid_file_path = vid_str
        self.folder_path = ""
        self.vid = cv2.VideoCapture(vid_str)

        self.fps = fps
        self.in_to_out_fps = int(self.vid.get(cv2.CAP_PROP_FPS) / fps)

        self.__get_path()

        # make folder for known faces
        if "known_faces_id" not in os.listdir(self.folder_path):
            path = os.path.join(self.folder_path, "known_faces_id")
            os.mkdir(path)
        self.known_faces_dir = f"{self.folder_path}/known_faces_id"
        self.__del_known_faces()

        self.video_data = []

        # Make class variables
        self.known_faces, self.known_names = [], []
        self.face_id = 0

        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=6, refine_landmarks=True,
                                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.cv_detector = cv2.QRCodeDetector()

    def analyze_video(self, frame_th=3, font_th=2, show_vid=True):
        """
        Uses Face, QR-name, and speaker recognition for each frame of the provided video
        - updates self.video_data with frame video data

        :param frame_th: Frame thickness uses for drawing rectanges if show_vid=True (int)
        :param font_th: Font thickness uses for face id if show_vid=True (int)
        :param show_vid: Shows frame by frame video during recognition (bool)
        :return: None
        """
        speaker_dir = owl_vid.__speaker_dir()

        ii = 0
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, ii)
        pbar = tqdm(desc='Processing Video', total=int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)))
        pbar.update(ii)

        talking = defaultdict(list)

        while True:
            ret, img = self.vid.read()
            if not ret:
                break

            if ii % self.in_to_out_fps:
                ii += 1
                continue

            frame = ii//self.in_to_out_fps
            time = frame / self.fps

            # Speaker dirization
            speaker = None
            while len(speaker_dir) > 0 and time > speaker_dir[0][1]:
                speaker_dir.pop(0)
            if len(speaker_dir) > 0 and speaker_dir[0][0] <= time <= speaker_dir[0][1]:
                speaker = speaker_dir[0][2]

            # Contour Locations
            contours = self.__calc_contours(img)

            # Face mesh
            mouths = self.__face_mesh(img, contours, talking)
            num_people = len(mouths)

            # QR Code Detection
            qr_values = self.__detect_qr_codes(img, contours)

            # Face Recognition
            miny = sorted(contours[:], key=lambda my: my[2])[0][1]
            matches, locations, tl, br, tl1, br1 = self.__face_recog(img, ii, miny)

            talker = None
            for contour_frame in talking.keys():
                for i, _ in enumerate(talking[contour_frame]):
                    talking[contour_frame][i][1] = max(0, talking[contour_frame][i][1]-1)
                    if talking[contour_frame][i][1] > 100:
                        talker = talking[contour_frame][i][0][:]

            if show_vid:
                for x, y, w, h in contours:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                for i, face_location, in enumerate(locations):
                    color = [131, 52, 235]
                    cv2.rectangle(img, tl[i], br[i], color, frame_th)
                    cv2.rectangle(img, tl1[i], br1[i], color, cv2.FILLED)
                    cv2.putText(img, matches[i], (face_location[3] + 10, face_location[2] + 15 + miny),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), font_th)

                for m_org, m in mouths:
                    cv2.putText(img, m, m_org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow('Image', img)
                cv2.waitKey(1)
                cv2.destroyAllWindows()

            self.video_data.append([frame, time, num_people, mouths, talker, matches, br, speaker])
            ii += 1
            pbar.update(self.in_to_out_fps)
        pbar.close()

    def analyze_vid_data(self):
        """
        Analyzes video data and matches speakers recognized using speaker dirization and matches them to a recognized
        face
        :return: Dictionary mapping speakers to face ids (dict)
        """
        speaker_dict = defaultdict(list)
        for i, data in enumerate(self.video_data):
            frame, time, num_people, mouths, talker, matches, locations, speaker = data
            # print(f"speakers: {speaker}, talker: {talker}, matches: {matches}, locations: {locations}")
            if speaker is not None and talker is not None and len(matches) > 0:
                # print(f"speakers: {speaker}, talker: {talker}, matches: {matches}, locations: {locations}")
                for match_idx, face_locations in enumerate(locations):
                    bottom_right = np.array([face_locations[0], face_locations[1]])
                    # print(f"distance for {match_idx}: {np.linalg.norm(bottom_right - talker)}")
                    if np.linalg.norm(bottom_right - talker) < 100.0:
                        speaker_dict[speaker].append(matches[match_idx])
        # print(speaker_dict)
        for s in speaker_dict.keys():
            speaker_dict[s] = max(set(speaker_dict[s]), key=speaker_dict[s].count)
        # print(speaker_dict)
        return speaker_dict

    def __face_recog(self, img, i, miny=0, model='hog', tol=0.55):
        """
        Calls the face_recognition library to recognize faces in a given video frame

        :param img: 2D Image Array (np array)
        :param i: Frame Number (int)
        :param miny: [OPTIONAL] Minimum y-idx to crop the video frame to - crops top OWL view if needed (int)
        :param model: Model to use for face recognition - set to "hog" (str)
        :param tol: Tolerance for face recognition library from 0 to 1 - set to 0.55 (float)
        :return: Matches, locations, top_left, bottom_right, top_left1, bottom_right1 (tuple)
                 Matches: Face ID recognized for the given frame
                 locations: Locations of faces recognized in the given frame (y axis adjusted)
                 top left / bottom right: Locations for drawing rectangles in frame if video is shown during processing
        """
        img = img.copy()[miny:]
        locations = fr.face_locations(img, model=model)
        encodings = fr.face_encodings(img, locations)
        matches = []

        top_left, top_left1, bottom_right, bottom_right1 = [], [], [], []
        for face_encoding, face_location, in zip(encodings, locations):
            results = fr.compare_faces(self.known_faces, face_encoding, tol)
            if True in results:
                # Get face id
                match = self.known_names[results.index(True)]
                matches.append(match)
            else:
                match = str(self.face_id)
                matches.append(match)
                self.face_id += 1
                self.known_names.append(match)
                self.known_faces.append(face_encoding)
                os.mkdir(f"{self.known_faces_dir}/{match}")
                pickle.dump(face_encoding, open(f"{self.known_faces_dir}/{match}/{match}--{i}.pkl", "wb"))
            top_left.append((face_location[3], face_location[0] + miny))
            bottom_right.append((face_location[1], face_location[2] + miny))
            top_left1.append((face_location[3], face_location[2] + miny))
            bottom_right1.append((face_location[1], face_location[2] + 22 + miny))

        return matches, locations, top_left, bottom_right, top_left1, bottom_right1

    def __face_mesh(self, img, contours, talking, show_image=False):
        """
        Uses Mediapipe's face mesh library to track mouth movements and recognize if any person is speaking (tallking)
        at any point in time. Analyzes each contour in a given frame.

        :param img: 2D Image Array (np array)
        :param contours: Locations of each contour to check for speakers - x, y, w, h for each idx (list)
        :param talking: List of talking scores for each face location - tuple of mouth origin, talking score, and previous
                        state of mouth (open/close) (tuple)
        :param show_image: Shows frame by frame video during recognition (bool)
        :return: List of updated talking scores for each face location tuple of mouth origin, talking score, and previous
                        state of mouth (open/close) (tuple)
        """
        mouths = []
        for contour_idx, (x, y, w, h) in enumerate(contours):
            image = img[y:y+h, x:x+w]
            face_mesh = self.face_mesh

            # need to improve image here for better processing accuracy
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            if show_image:
                cv2.imshow('Image', image)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
            if results.multi_face_landmarks:
                for faces in results.multi_face_landmarks:
                    dist = 0
                    lm = faces.landmark
                    face_dist = 1 / np.linalg.norm((np.array([lm[10].x, lm[10].y]) - np.array([lm[152].x, lm[152].y])))
                    for u, l in [(13, 14), (82, 82), (312, 312)]:
                        dist += np.linalg.norm((np.array([lm[u].x, lm[u].y]) - np.array([lm[l].x, lm[l].y]))) * 10000
                    dist = (face_dist * dist) / 3
                    mouth = "closed"
                    if dist > 40:
                        mouth = "open"
                    x_mouth, y_mouth = faces.landmark[375].x, faces.landmark[375].y
                    mouth_text_org = np.add(np.multiply([x_mouth, y_mouth], [w, h]), [x+10, y]).astype(int)

                    if contour_idx not in talking.keys():
                        talking[contour_idx].append([mouth_text_org, 3, mouth])
                    else:
                        mouth_close = False
                        for i, m in enumerate(talking[contour_idx]):
                            mouth_dist = np.linalg.norm(m[0] - mouth_text_org)
                            talking_score, mouth_prev = m[1], m[2]
                            if mouth_dist < 50.0:
                                mouth_close = True
                                talking[contour_idx][i][0] = mouth_text_org[:]
                                if mouth_prev == "closed" and mouth == "open":
                                    talking[contour_idx][i][1] = min(135, talking_score + 40)
                                talking[contour_idx][i][2] = mouth[:]
                                break
                        if not mouth_close:
                            talking[contour_idx].append([mouth_text_org, 0, mouth])
                    mouths.append((mouth_text_org, mouth))
            else:
                if contour_idx in talking.keys():
                    del talking[contour_idx]

        return mouths

    def __detect_qr_codes(self, image, contours):
        """
        Uses CV2 QR code detector to read QR codes in each contour of a given frame
        :param image: 2D Image Array (np array)
        :param contours: Locations of each contour to check for speakers - x, y, w, h for each idx (list)
        :return: QR code values found for a given frame
        """
        qr_values = []
        for x, y, w, h in contours:
            try:
                img = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                data, bbox, _, _ = self.cv_detector.detectAndDecodeMulti(img)
            except ValueError:
                data = ""
                pass
            qr_values.append(data)
        return qr_values

    @staticmethod
    def __detect_qr_pyboof(image, contours):
        """
        Uses PyBoof QR code detector to read QR codes in each contour of a given frame
        :param image: 2D Image Array (np array)
        :param contours: Locations of each contour to check for speakers - x, y, w, h for each idx (list)
        :return: QR code values found for a given frame
        """
        qr_values = []
        boof_detector = pb.FactoryFiducial(np.uint8).microqr()
        for x, y, w, h in contours:
            image = pb.ndarray_to_boof(image[y:y+h, x:x+w])
            boof_detector.detect(image)
            for qr in boof_detector.detections:
                qr_values.append((qr.message, qr.bounds.convert_tuple()))

    def __speaker_dir(self):
        """
        Uses pyannote.audio speaker dirization to recognize and identify different speakers at different points in time
        :return: List of speaker identification numbers and speaking time start and end (list)
        """
        audio = "temp_audio.wav"
        command = f"ffmpeg -i ./{self.vid_file_path} -ab 160k -ac 2 -ar 44100 -vn ./{self.folder_path}/{audio}"
        subprocess.call(command, shell=True)

        print("Getting Pipeline")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

        print("Diarization")
        diarization = pipeline(f"{self.folder_path}/{audio}")

        speaker_dir = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            speaker_dir.append([turn.start, turn.end, speaker])

        os.remove(f"{self.folder_path}/{audio}")
        return speaker_dir

    @staticmethod
    def __calc_contours(image):
        """
        Finds all contours (speaker view) of all speakers in a given frame
        :param image: 2D Image Array (np array)
        :return: Locations of each contour to check for speakers - x, y, w, h for each idx (list)
        """
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY_INV)

        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        min_area = 500 * 500
        new_contours = []

        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area:
                M = cv2.moments(c)
                cy = int(M['m01'] / M['m00'])
                if cy > 200:
                    epsilon = 0.08 * cv2.arcLength(c, True)
                    c = cv2.approxPolyDP(c, epsilon, True)
                    new_contours.append(cv2.boundingRect(c))
        new_contours = sorted(new_contours, key=lambda x: x[0])
        return new_contours

    def __get_path(self):
        """
        Gets path to the input video file used for analysis
        :return: Path to folder with video file (str)
        """
        path_split = self.vid_file_path.split("/")
        path_split.pop(-1)
        # self.filename = path_split.pop(-1)
        self.folder_path = "/".join(path_split)

    def __del_known_faces(self):
        """
        Deletes the temporary folder created "/known_faces_id" and all face encodings in the self.folder_path folder
        :return: None
        """
        known_faces_path = f"{self.folder_path}/known_faces_id"
        for name in os.listdir(known_faces_path):
            if name.isdigit():
                shutil.rmtree(f"{known_faces_path}/{name}")
        return

