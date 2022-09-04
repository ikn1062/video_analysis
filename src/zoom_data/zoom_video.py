import cv2
import face_recognition as fr
import numpy as np
import pytesseract
import os
import time
import pickle
import csv
import shutil
import threading
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import KMeans
import datetime
from tqdm import tqdm
from fuzzywuzzy import process


class ZoomVideo:
    class Decorators(object):
        @classmethod
        def zoom_decorator(cls, func):
            """
            Creates and starts a thread for a given function
            :param func: function object
            :return: thread object for function
            """
            def zoom_thread(*args):
                t = threading.Thread(target=func, args=args)
                t.start()
                return t
            return zoom_thread

    def __init__(self, video_path, results_path, num_participants=None, participants=None, db=None, fps=1):
        """
        Analyzes Zoom Video to extract data on who is speaking at what time
        - Cleans video data and extracts video frame data to csv and accuracy of analysis to txt in results folder
        - Uploads data to MySQL database if one is provided
        :param video_path: Path to video file (str)
        :param results_path: Path to results folder (str)
        :param num_participants: [OPTIONAL] Number of participants in the video - used for clustering (int)
        :param participants: [OPTIONAL] List of participants for the video - used for fuzzy-matching of names (list)
        :param db: [OPTIONAL] Database (MySQL Database object)
        :param fps: [OPTIONAL] FPS to analyze video with - set to 1 fps (float)
        """
        self.filename = ""
        self.folder_path = ""
        self.video_file_str = video_path
        self.results_path = results_path
        self.cropped_video_str = self.video_file_str[:-4] + "_cropped.mp4"
        self.__get_path()

        self.fps = fps
        self.num_participants = num_participants
        self.participants = participants
        self.video_data = []
        self.row_vec_arr = []

        self.control_flag = False

        self.locations, self.encodings = [], []

        self.name, self.name_id = "Unknown", 0
        self.img_row_vec = []

        self.db = db

    def analyze_video(self):
        """
        Analyzes input video and creates output in the results path - calls all the methods needed for video analysis
        :return: None
        """
        print(self.filename)
        self.crop_video()
        self.recognition()
        self.process_data()
        self.del_known_faces()
        os.remove(self.cropped_video_str)
        return 0

    def crop_video(self, show_cropped_video=False):
        """
        Crops video, lowers frame resolution and frame rate for faster video analysis
        - Creates a new cropped video in the video folder
        :param show_cropped_video: Whether to show the output cropped video during crop processing (bool)
        :return: None
        """
        # Define output file string
        video_file_str = self.video_file_str

        cap = cv2.VideoCapture(video_file_str)

        # Setting Cropped dimensions for video
        x, y, w, h = self.__corner_detection()
        # Define the codec and create VideoWriter object
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.cropped_video_str, fourcc, self.fps, (w, h))  # might need to change

        # Crops video to specified dimensions
        index_in, index_out = -1, -1
        i = 0
        pbar = tqdm(desc='Cropping Video', total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        while True:
            if not cap.grab():
                break
            index_in += 1

            out_due = int(index_in / fps_in * self.fps)
            if out_due > index_out:
                success, frame = cap.retrieve()
                if not success:
                    break
                index_out += 1

                if not i % 5 and not self.check_controls_panel(frame):
                    self.control_flag = True

                crop_img = frame[y:y + h, x:x + w, :]
                crop_img = cv2.resize(crop_img, (w, h))
                out.write(crop_img)
                i += 1
                if show_cropped_video:
                    cv2.imshow("Video", crop_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            pbar.update(self.fps)
        pbar.close()
        cap.release()
        out.release()

    def recognition(self, tol=0.55, show_vid=False):
        """
        Uses Face, name, and image-row recognition for each frame of the provided video
        - updates self.video_data with frame video data
        :param tol: Face recognition tolerance from 0 to 1 (float)
        :param show_vid: Shows frame by frame video during recognition
        :return: None
        """
        if "known_faces_id" not in os.listdir(self.folder_path):
            path = os.path.join(self.folder_path, "known_faces_id")
            os.mkdir(path)
        known_faces_dir = f"{self.folder_path}/known_faces_id"

        frame_th, font_th = 3, 1

        video = cv2.VideoCapture(self.cropped_video_str)
        video2 = cv2.VideoCapture(self.video_file_str)

        row_vec_arr = self.row_vec_arr

        # Set up recognition of known faces and known names, and frame count
        known_faces, known_names = [], []
        face_id, i = 0, 0
        pbar = tqdm(desc='Video Recognition', total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
        while True:
            ret, image = video.read()
            if not ret:
                break

            # Collect face location and encoding
            t_face = self.face_recog(image)

            # Find name from image and create img, row vector
            t_name = self.__get_name(image, video2, i)

            # Convert current image to vector
            t_row = self.__img_row_avg(image)

            t_name.join()
            t_row.join()
            t_face.join()

            name = self.name
            img_row_vec = self.img_row_vec

            match = "Unknown"
            for face_encoding, face_location, in zip(self.encodings, self.locations):
                # Get face result matching from comparisons
                results = fr.compare_faces(known_faces, face_encoding, tol)

                if True in results:
                    # Get face id
                    match = known_names[results.index(True)]

                    # Match the image vectors to faces
                    if i - row_vec_arr[int(match)][0] > 30:
                        row_vec_arr[int(match)][0] = i
                        row_vec_arr[int(match)][1].append(img_row_vec)

                else:
                    # Match id is added to name_dict
                    match = str(face_id)

                    # add 1d image vector to array
                    row_vec_arr.append([i, [img_row_vec]])
                    # iterate id
                    face_id += 1

                    # Add face to known_names and known_faces, encode and save face in files
                    known_names.append(match)
                    known_faces.append(face_encoding)
                    os.mkdir(f"{known_faces_dir}/{match}")
                    pickle.dump(face_encoding, open(f"{known_faces_dir}/{match}/{match}--{int(time.time())}.pkl", "wb"))

                if show_vid:
                    # Add overlay for face when showing vid capture
                    top_left, bottom_right = (face_location[3], face_location[0]), (face_location[1], face_location[2])
                    color = [131, 52, 235]
                    cv2.rectangle(image, top_left, bottom_right, color, frame_th)
                    top_left_1 = (face_location[3], face_location[2])
                    bottom_right_1 = (face_location[1], face_location[2] + 22)
                    cv2.rectangle(image, top_left_1, bottom_right_1, color, cv2.FILLED)
                    cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), font_th)

            self.video_data.append([name, match, img_row_vec])

            i += 1
            pbar.update(1)
            if show_vid:
                cv2.imshow("Video", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        pbar.close()
        cv2.destroyAllWindows()
        video.release()

    def process_data(self):
        """
        Processes Frame Video Data (self.video_data)
        - Cleans frame video data and exports data to csv file (uploads to db if db_info provided)
        :return: None
        """
        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        results_name = f"{self.results_path}/{self.filename}_{now_time}"

        if self.db:
            print(self.filename)
            self.db.create_video_table(self.filename[:-4])

        if self.participants:
            name_list = self.__fuzz_name()
            name_label = "fuzz score"
        else:
            name_list = self.__name_cluster()
            name_label = "name id"

        # Open CSV writer
        vid_data_csv = open(f"{results_name}.csv", 'w', newline="")
        writer = csv.writer(vid_data_csv)

        name_known_count, face_known_count, row_count = 0, 0, 0
        writer.writerow(["frame", "time", "name", name_label, "face id", "row id"])

        pbar = tqdm(desc='Process Video Data', total=len(self.video_data))
        dt = 1 / self.fps
        ii = 0
        for name, match, img_row_vector in tqdm(self.video_data, desc='Processing Video Data'):
            name, name_id = name_list[ii][0], name_list[ii][1]
            row_idx = self.compare_img_vector(self.row_vec_arr, img_row_vector, tresh=15000, norm=cv2.NORM_L1)

            if name != "Unknown":
                name_known_count += 1
            if match != "Unknown":
                face_known_count += 1
            if row_idx != -1:
                row_count += 1

            vid_dat = [ii, ii * dt, name, name_id, match, row_idx]
            writer.writerow(vid_dat)

            if self.db:
                self.db.insert_video_data(self.filename[:-4], tuple(vid_dat))

            ii += 1
            pbar.update(1)

        vid_data_csv.close()
        pbar.close()

        if self.db:
            acc_data = (name_known_count, face_known_count, row_count)
            self.db.insert_accuracy_data(self.filename[:-4], acc_data)
            self.db.commit()

        with open(f"{results_name}.txt", 'w') as t:
            t.write(f"ratio of pytesseract name detection: {name_known_count / ii}\n")
            t.write(f"ratio of face recog name detection: {face_known_count / ii}\n")
            t.write(f"ratio of any name found detection: {row_count / ii}\n")

    def __name_recognition(self, image, show_name=False):
        """
        Uses Pytesseract OCR to recognize the speaker in a given image frame
        :param image: 2D Image Array (np array)
        :param show_name: Whether to show name for a given frame (bool)
        :return: Name recognized (str)
        """
        image = image[450:, :300]
        image = self.__preprocess_img_name_recog(image)
        name = pytesseract.image_to_string(image)
        if not name:
            name = "Unknown"

        if show_name:
            print("name recognized: " + str(name))

        return name

    @staticmethod
    def __preprocess_img_name_recog(image):
        """
        Pre-processes image before name recognition
        :param image: 2D Image Array (np array)
        :return: Processed 2D image array (np array)
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        # image = cv2.medianBlur(image, 3, 51)
        image = cv2.GaussianBlur(image, (7, 7), 0)
        image = cv2.Canny(image=image, threshold1=160, threshold2=210)
        # image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return image

    def __name_cluster(self):
        """
        Assigns names to self.num_participant number of clusters name ids
        - Uses k-means clustering methods to find what names match with each other
        - Used if self.participants is not provided
        :return: List of names and the cluster identification (list)
        """
        clean_name_list = self.__clean_names()

        name_cluster = []

        vectorizer = HashingVectorizer(n_features=2 ** 2, lowercase=True, analyzer='char')
        name_vec = vectorizer.fit_transform(clean_name_list)
        name_arr = name_vec.toarray()
        if not self.num_participants:
            self.num_participants = len(self.row_vec_arr)
        model = KMeans(n_clusters=self.num_participants).fit_predict(name_arr)

        for i, n in enumerate(clean_name_list):
            name_cluster.append([n, model[i]])
        return name_cluster

    def __fuzz_name(self):
        """
        Uses fuzzy matching to compare detected name in a given frame to a list of participant names in self.participants
        :return: List of fuzzymatched name for each frame
        """
        clean_name_list = self.__clean_names()
        fuzz_names = []
        for name in clean_name_list:
            if len(name) < 0:
                fuzz_names.append(["Unknown", 0])
                continue
            fuzz_name, fuzz_score = process.extractOne(name, self.participants)
            if fuzz_score < 50:
                fuzz_name = "Unknown"
            fuzz_names.append([fuzz_name, fuzz_score])
        return fuzz_names

    def __clean_names(self):
        """
        Cleans the OCR name output in self.video_data for each frame
        :return: list of cleaned names for each frame (list)
        """
        name_list = np.array(self.video_data)
        name_list = name_list[:, 0]
        clean_name_list = []
        for i, row in enumerate(name_list):
            name = max(row.split('\n'), key=len)
            clean_name_list.append(name)
        return clean_name_list

    def __corner_detection(self):
        """
        Detects the corners on a given frame to crop the video to only the speaker view
        :return: Contour of the speaker - top-left x, top-left y, width, height (tuple)
        """
        frame = 0
        vid = cv2.VideoCapture(self.video_file_str)
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
        success, img = vid.read()

        pbar = tqdm(desc='Contour Recognition', total=int(vid.get(cv2.CAP_PROP_FRAME_COUNT)))
        while success and not self.check_controls_panel(img, tresh=11000):
            frame += self.fps
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
            success, img = vid.read()
            pbar.update(self.fps)
        pbar.close()

        if not success:
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, img = vid.read()

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        min_area, max_area = 400*400, 0
        contour = None
        # result = img.copy()
        for c in contours:
            area = cv2.contourArea(c)
            # cv2.drawContours(result, [c], -1, (0, 255, 0), 1)
            if area > min_area and area > max_area:
                epsilon = 0.08 * cv2.arcLength(c, True)
                c = cv2.approxPolyDP(c, epsilon, True)
                # cv2.drawContours(result, [c], -1, (0, 0, 255), 1)
                contour = cv2.boundingRect(c)

        # show images
        # cv2.imshow("RESULT", result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        x, y, w, h, = contour
        vid.release()
        return x, y, w, h

    def del_known_faces(self):
        """
        Deletes the temporary folder created "/known_faces_id" and all face encodings in the self.folder_path folder
        :return: None
        """
        known_faces_path = f"{self.folder_path}/known_faces_id"
        for name in os.listdir(known_faces_path):
            if name.isdigit():
                shutil.rmtree(f"{known_faces_path}/{name}")
        return 0

    @Decorators.zoom_decorator
    def face_recog(self, image, model="hog"):
        """
        Runs facial recognition in the current frame of the video, using "hog" model
        - Assigns face locations to self.locations and face encodings to self.encodings
        :param image: 2D Image Array (np array)
        :param model: Face Recognition ML model to use (str)
        :return: None
        """
        self.locations = fr.face_locations(image, model=model)
        self.encodings = fr.face_encodings(image, self.locations)

    @Decorators.zoom_decorator
    def __get_name(self, image, video2, i):
        """
        Gets the name of the speaker in the current frame of the video
        - Outputs the name of the speaker in self.name
        :param image: 2D Image Array (np array)
        :param video2: Un-cropped Video (cv2.VideoCapture object)
        :param i: Frame Number (int)
        :return: None
        """
        self.name = self.__name_recognition(image)
        if self.control_flag and self.name == "Unknown":
            vid_fps = video2.get(cv2.CAP_PROP_FPS)
            video2.set(cv2.CAP_PROP_POS_FRAMES, i * int(vid_fps / self.fps))
            success, img2 = video2.read()
            if success and not self.check_controls_panel(img2, 14500):
                self.name = self.__name_recognition(img2)

    @Decorators.zoom_decorator
    def __img_row_avg(self, image):
        """
        Creates a 1D row vector of the input image and stores it in self.img_row_vec
        :param image: 2D Image Array (np array)
        :return: None
        """
        img_vec = np.array([])
        for i in range(len(image)):
            img_vec = np.append(img_vec, image[i].mean(axis=0))
        self.img_row_vec = img_vec

    def __get_path(self):
        """
        Gets path to the input video file used for analysis
        :return: Path to folder with video file (str)
        """
        path_split = self.video_file_str.split("/")
        self.filename = path_split.pop(-1)
        self.folder_path = "/".join(path_split)
        print(self.folder_path)

    def check_controls_panel(self, img, tresh=20000, norm=cv2.NORM_L2):
        """
        Checks if a control panel exists in a given frame
        :param img: 2D Image Array to check for a control panel (np array)
        :param tresh: Maximum distance threshold for identification (int)
        :param norm: Norm method to calculate distance between two image arrays (cv2.NORM object)
        :return: If a control panel exists in a given frame (bool)
        """
        leave_room = cv2.imread("data/leave_room_box.png")
        dist = cv2.norm(self.oned_image(leave_room), self.oned_image(img[-100:, -150:]), norm)
        if dist > tresh:
            # No control panel
            return False
        # control panel
        return True

    @staticmethod
    def oned_image(image):
        """
        Creates a 1D np array of the input image
        :param image: Image to be converted to 1D
        :return: Image reshaped as 1D Array
        """
        rows, cols, colors = image.shape
        return image.reshape(rows * cols * colors)

    @staticmethod
    def compare_img_vector(img_vector_arr, img_1d_vector, thresh=100000, norm=cv2.NORM_L2):
        """
        Compares an image vector (img_1d_vector) to an array of image vectors, to find the closest distance below
        a given threshold
        :param img_vector_arr: List of 1D image vectors (list)
        :param img_1d_vector: 1D image vector to compare img_vector_arr to (np array)
        :param thresh: Maximum distance threshold for identification (int)
        :param norm: Norm method to calculate distance between two 1D image vectors (cv2.NORM object)
        :return: image index of closest distance - -1 if all distances above threshold (int)
        """
        img_idx, min_dist = -1, 2e31
        for ii, (_, img_vectors) in enumerate(img_vector_arr):
            total_dist = 0
            for img_vec in img_vectors:
                total_dist += cv2.norm(img_1d_vector, img_vec, norm)
            curr_dist = total_dist / len(img_vectors)
            if curr_dist < thresh and curr_dist < min_dist:
                img_idx = ii
                min_dist = curr_dist + 0

        if img_idx == -1:
            return -1
        return img_idx
