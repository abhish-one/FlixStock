import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from keras.optimizers import Adam
from keras.utils import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from Flixstock import Flix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import pandas as pd

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-t", "--patternbin", required=True,
	help="path to output pattern label binarizer")
ap.add_argument("-s", "--sleevebin", required=True,
	help="path to output sleeve_length label binarizer")
ap.add_argument("-n", "--neckbin", required=True,
	help="path to output neck label binarizer")
ap.add_argument("-p", "--plot", type=str, default="output",
	help="base filename for generated plots")
args = vars(ap.parse_args())

print("info : Loading images. . . ")
df = pd.read_csv('data/attributes.csv')
df = df.fillna(-1)
data = []
neckLabels = []
sleeveLabels = []
patternLabels = []
imagepath = "data/images"

for i in range(df.shape[0]):
    try:
        image = cv2.imread(imagepath+"\\"+str(df.iloc[i]['filename']))
        image = cv2.resize(image,(96,96))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = img_to_array(image)
        data.append(image)
        neckLabels.append(df.iloc[i]['neck'])
        sleeveLabels.append(df.iloc[i]['sleeve_length'])
        patternLabels.append(df.iloc[i]['pattern'])
    except:
        continue
    
#scale raw pixel intensities to (0,1) and covert it into numpy array
data = np.array(data,dtype='float')/255.0 
print("[INFO] "+str(data.shape[0])+" images has been loaded")
#convert label lists to numpy array
neckLabels = np.array(neckLabels)
sleeveLabels = np.array(sleeveLabels)
patternLabels = np.array(patternLabels)

df2 = pd.DataFrame({'neck':neckLabels,'sleeve_length':sleeveLabels,'pattern':patternLabels})
df2.to_csv('temp_labels.csv')
# binarize the labels
print("[INFO] binarizing Labels. . .")
neckLB = LabelBinarizer()
sleeveLB = LabelBinarizer()
patternLB = LabelBinarizer()
neckLabels = neckLB.fit_transform(neckLabels)
sleeveLabels = sleeveLB.fit_transform(sleeveLabels)
patternLabels = patternLB.fit_transform(patternLabels)


split = train_test_split(data,neckLabels,sleeveLabels,patternLabels, test_size=0.2,random_state=42)
(trainX,testX,trainNeckY,testNeckY,trainSleeveY,testSleeveY,trainPatternY,testPatternY) = split

# initialize our FlixStock multi-output network
model = Flix.build(96, 96,
	numNeck=len(neckLB.classes_),
	numSleeve=len(sleeveLB.classes_),
    numPattern=len(patternLB.classes_),
	finalAct="softmax")
# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
	"neck_output": "categorical_crossentropy",
	"sleeve_output": "categorical_crossentropy",
    "pattern_output": "categorical_crossentropy"
}
lossWeights = {"neck_output": 1.0, "sleeve_output": 1.0,"pattern_output": 1.0}
# initialize the optimizer and compile the model
print("[INFO] compiling model...")

INIT_LR = 1e-3
EPOCHS = 70
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])

# train the network to perform multi-output classification
H = model.fit(x=trainX,
	y={"neck_output": trainNeckY, "sleeve_output": trainSleeveY, "pattern_output": trainPatternY},
	validation_data=(testX,
		{"neck_output": testNeckY, "sleeve_output": testSleeveY,"pattern_output": testPatternY}),
	epochs=EPOCHS,
	verbose=1)
# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"], save_format="h5")

# save the pattern binarizer to disk
print("[INFO] serializing pattern label binarizer...")
f = open(args["patternbin"], "wb")
f.write(pickle.dumps(patternLabels))
f.close()
# save the neck binarizer to disk
print("[INFO] serializing neck label binarizer...")
f = open(args["neckbin"], "wb")
f.write(pickle.dumps(neckLabels))
f.close()
# save the sleeve binarizer to disk
print("[INFO] serializing sleeve label binarizer...")
f = open(args["sleevebin"], "wb")
f.write(pickle.dumps(sleeveLabels))
f.close()


# plot the total loss, pattern loss, neck loss and sleeve loss
lossNames = ["loss", "pattern_output_loss", "neck_output_loss","sleeve_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(4, 1, figsize=(13, 13))
# loop over the loss names
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
		label="val_" + l)
	ax[i].legend()
# save the losses figure
plt.tight_layout()
plt.savefig("{}_losses.png".format(args["plot"]))
plt.close()
# create a new figure for the accuracies
accuracyNames = ["pattern_output_accuracy", "neck_output_accuracy","sleeve_output_accuracy"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(8, 8))
# loop over the accuracy names
for (i, l) in enumerate(accuracyNames):
	# plot the loss for both the training and validation data
	ax[i].set_title("Accuracy for {}".format(l))
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Accuracy")
	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
		label="val_" + l)
	ax[i].legend()
# save the accuracies figure
plt.tight_layout()
plt.savefig("{}_accs.png".format(args["plot"]))
plt.close()
