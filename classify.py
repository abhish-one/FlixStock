from keras.utils import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import cv2
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--patternbin", required=True,
	help="path to output pattern label binarizer")
ap.add_argument("-n", "--neckbin", required=True,
	help="path to output neck label binarizer")
ap.add_argument("-s", "--sleevebin", required=True,
	help="path to output sleeve label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
output = imutils.resize(image, width=400)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


# load the trained convolutional neural network from disk, followed
# by the pattern, neck and sleeve label binarizers, respectively
print("[INFO] loading network...")
model = load_model(args["model"], custom_objects={"tf": tf})
#patternLB = pickle.loads(open(args["patternbin"], "rb").read())
#neckLB = pickle.loads(open(args["neckbin"], "rb").read())
#sleeveLB = pickle.loads(open(args["sleevebin"], "rb").read())
df = pd.read_csv('temp_labels.csv')
patternLabels = df['pattern']
neckLabels = df['neck']
sleeveLabels = df['sleeve_length']

patternLB = LabelBinarizer()
neckLB = LabelBinarizer()
sleeveLB = LabelBinarizer()

neckLabels = neckLB.fit_transform(neckLabels)
patternLabels = patternLB.fit_transform(patternLabels)
sleeveLabels = sleeveLB.fit_transform(sleeveLabels)

# classify the input image using Keras' multi-output functionality
print("[INFO] classifying image...")
(neckProba, sleeveProba, patternProba) = model.predict(image)
# find indexes of pattern, neck and sleeve outputs with the
# largest probabilities, then determine the corresponding class
# labels

patternIdx = patternProba[0].argmax()
neckIdx = neckProba[0].argmax()
sleeveIdx = sleeveProba[0].argmax()
patternLabel = patternLB.inverse_transform(patternLabels)[patternIdx]
neckLabel = neckLB.inverse_transform(neckLabels)[neckIdx]
sleeveLabel = sleeveLB.inverse_transform(sleeveLabels)[sleeveIdx]

# draw the pattern, neck and sleeve_length label on the image
patternText = "pattern: {} ({:.2f}%)".format(patternLabel,
	patternProba[0][patternIdx] * 100)
neckText = "neck: {} ({:.2f}%)".format(neckLabel,
	neckProba[0][neckIdx] * 100)
sleeveText = "sleeve_length: {} ({:.2f}%)".format(sleeveLabel,
	sleeveProba[0][sleeveIdx] * 100)
cv2.putText(output, patternText, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
cv2.putText(output, neckText, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
cv2.putText(output, sleeveText, (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
# display the predictions to the terminal as well
print("[INFO] {}".format(patternText))
print("[INFO] {}".format(neckText))
print("[INFO] {}".format(sleeveText))
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)