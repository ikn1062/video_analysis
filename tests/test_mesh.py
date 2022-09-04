import mediapipe as mp
import cv2
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

"""
https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
  lipsUpperOuter: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
  lipsLowerOuter: [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
  lipsUpperInner: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
  lipsLowerInner: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
"""


def main():
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        talking = 0
        mouth = ""
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            x_mouth, y_mouth = 0, 0
            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for faces in results.multi_face_landmarks:
                    dist = 0
                    lm = faces.landmark
                    face_dist = 1 / np.linalg.norm((np.array([lm[10].x, lm[10].y]) - np.array([lm[152].x, lm[152].y])))
                    for u, l in [(13, 14), (82, 82), (312, 312)]:
                        # scale based on how large the face is?
                        dist += np.linalg.norm((np.array([lm[u].x, lm[u].y]) - np.array([lm[l].x, lm[l].y]))) * 10000
                    dist = (face_dist * dist) / 3
                    print(dist)
                    mouth_prev = mouth[:]
                    if dist > 40:
                        mouth = "open"
                    else:
                        mouth = "closed"
                    if mouth_prev == "closed" and mouth == "open":
                        talking = min(110, talking+35)
                    x_mouth = faces.landmark[375].x
                    y_mouth = faces.landmark[375].y
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                # Flip the image horizontally for a selfie-view display.
            talking = max(0, talking-1.5)
            if talking >= 50:
                talking_origin = tuple(np.add(np.multiply([x_mouth, y_mouth], [1920, 1080]), [10, 30]).astype(int))
                cv2.putText(image, "talking", talking_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            mouth_text_org = tuple(np.add(np.multiply([x_mouth, y_mouth], [1920, 1080]), [10, 0]).astype(int))
            cv2.putText(image, mouth, mouth_text_org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('MediaPipe Face Mesh', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()


if __name__ == "__main__":
    main()
