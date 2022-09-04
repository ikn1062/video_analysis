import cv2
import numpy as np
import pyboof as pb
import time
from tqdm import tqdm

# Read image
filenames = ["data/owl-data/qr1.png", "data/owl-data/qr2.png", "data/owl-data/qr3.png"]

for i in tqdm(range(0, 16)):
    for j in tqdm(range(10, 26)):
        for filename in filenames:
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image = cv2.GaussianBlur(image, (5, 5), 0)
            # kernel = np.array([[0, -1, 0],
            #                    [-1, 5, -1],
            #                    [0, -1, 0]])
            # image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
            _, image = cv2.threshold(image, i*11, j*11, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            qrCodeDetector = cv2.QRCodeDetector()
            decodedText, points, _ = qrCodeDetector.detectAndDecode(image)
            if len(decodedText) < 1:
                decodedText = None
            else:
                print(decodedText)

            original = pb.ndarray_to_boof(image)
            # pb.swing.show(original, title="Outputs")
            # input("Press any key to exit")

            detector = pb.FactoryFiducial(np.uint8).microqr()
            detector.detect(original)

            if len(detector.detections) > 1:
                print("Detected a total of {} QR Codes".format(len(detector.detections)))
                for qr in detector.detections:
                    print("Message: " + qr.message)
                    print("     at: " + str(qr.bounds))

            time.sleep(0.05)
