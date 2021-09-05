import cv2
import numpy as np
from skimage import measure
from imutils import perspective
import imutils

from src.data_utils import order_points, convert2Square, draw_labels_and_boxes
from src.lp_detection.detect import detectNumberPlate
from src.char_classification.model import CNN_Model
from skimage.filters import threshold_local

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

LP_DETECTION_CFG = {
    "weight_path": "./src/weights/yolov3-tiny_15000.weights",
    "classes_path": "./src/lp_detection/cfg/yolo.names",
    "config_path": "./src/lp_detection/cfg/yolov3-tiny.cfg"
}

CHAR_CLASSIFICATION_WEIGHTS = './src/weights/weight.h5'

class E2E(object):
    def __init__(self):
        self.image = np.empty((28, 28, 1))
        self.detectLP = detectNumberPlate(LP_DETECTION_CFG['classes_path'], LP_DETECTION_CFG['config_path'], LP_DETECTION_CFG['weight_path'])
        self.recogChar = CNN_Model(trainable=False).model
        self.recogChar.load_weights(CHAR_CLASSIFICATION_WEIGHTS)
        self.candidates = []

    def extractLP(self):
        coordinates = self.detectLP.detect(self.image)
        if len(coordinates) == 0:
            ValueError('No images detected')

        for coordinate in coordinates:
            yield coordinate

    def predict(self, image):
        # Input image or frame
        self.image = image

        for coordinate in self.extractLP():     # detect license plate by yolov3
            self.candidates = []

            # convert (x_min, y_min, width, height) to coordinate(top left, top right, bottom left, bottom right)
            pts = order_points(coordinate)

            # crop number plate used by bird's eyes view transformation
            LpRegion = perspective.four_point_transform(self.image, pts)
           
            # segmentation
            self.segmentation(LpRegion)

            # recognize characters
            self.recognizeChar()

            # format and display license plate
            license_plate = self.format()

            # draw labels
            self.image = draw_labels_and_boxes(self.image, license_plate, coordinate)

        return self.image

    def segmentation(self, LpRegion):
        # apply thresh to extracted licences plate
        V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]

        # adaptive threshold
        T = threshold_local(V, 15, offset=10, method="gaussian")
        thresh = (V > T).astype("uint8") * 255

        # convert black pixel of digits to white pixel
        thresh = cv2.bitwise_not(thresh)
        thresh = imutils.resize(thresh, width=400)
        thresh = cv2.medianBlur(thresh, 5)

        # connected components analysis
        labels = measure.label(thresh, connectivity=2, background=0)

        # loop over the unique components
        for label in np.unique(labels):
            # if this is background label, ignore it
            if label == 0:
                continue

            # init mask to store the location of the character candidates
            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255

            # find contours from mask
            _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(contour)

                # rule to determine characters
                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                heightRatio = h / float(LpRegion.shape[0])

                if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0:
                    # extract characters
                    candidate = np.array(mask[y:y + h, x:x + w])
                    square_candidate = convert2Square(candidate)
                    square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                    square_candidate = square_candidate.reshape((28, 28, 1))
                    self.candidates.append((square_candidate, (y, x)))

    def recognizeChar(self):
        characters = []
        coordinates = []

        for char, coordinate in self.candidates:
            characters.append(char)
            coordinates.append(coordinate)

        characters = np.array(characters)
        result = self.recogChar.predict_on_batch(characters)
        result_idx = np.argmax(result, axis=1)

        self.candidates = []
        for i in range(len(result_idx)):
            if result_idx[i] == 31:    # if is background or noise, ignore it
                continue
            self.candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))

    def format(self):
        first_line = []
        second_line = []

        for candidate, coordinate in self.candidates:
            if self.candidates[0][1][0] + 40 > coordinate[0]:
                first_line.append((candidate, coordinate[1]))
            else:
                second_line.append((candidate, coordinate[1]))

        def take_second(s):
            return s[1]

        first_line = sorted(first_line, key=take_second)
        second_line = sorted(second_line, key=take_second)

        if len(second_line) == 0:  # if license plate has 1 line
            license_plate = "".join([str(ele[0]) for ele in first_line])
        else:   # if license plate has 2 lines
            license_plate = "".join([str(ele[0]) for ele in first_line]) + "-" + "".join([str(ele[0]) for ele in second_line])

        return license_plate
