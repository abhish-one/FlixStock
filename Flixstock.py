from keras.models import Model
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Flatten
#from keras.applications import VGG19
from keras.layers import Input
import tensorflow as tf


class Flix:

    @staticmethod
    def default_hidden_layers(x,chanDim):
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3,3))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        return x

    @staticmethod
    def build_pattern_branch(inputs, numCategories, finalAct = "softmax",chanDim = -1):
        x = Lambda (lambda c: tf.image.rgb_to_grayscale(c))(inputs)  #converting RGB to grayscale 
        
        x = Flix.default_hidden_layers(x,chanDim)
        x = Flatten()(x)
        x = Dense(256)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(numCategories)(x)
        x = Activation(finalAct, name="pattern_output")(x)

        return x

    @staticmethod
    def build_sleeve_branch(inputs, numCategories, finalAct="softmax",chanDim=-1):
        x = Lambda (lambda c: tf.image.rgb_to_grayscale(c))(inputs)  #converting RGB to grayscale
        
        x = Flix.default_hidden_layers(x,chanDim)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(numCategories)(x)
        x = Activation(finalAct, name="sleeve_output")(x)
        # return the sleeve_length prediction sub-network
        return x

    @staticmethod
    def build_neck_branch(inputs, numCategories, finalAct="softmax",chanDim=-1):
        x = Lambda (lambda c: tf.image.rgb_to_grayscale(c))(inputs)  #converting RGB to grayscale
        
        x = Flix.default_hidden_layers(x,chanDim)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(numCategories)(x)
        x = Activation(finalAct, name="neck_output")(x)
        # return the neck prediction sub-network
        return x

    @staticmethod
    def build(width, height, numPattern, numSleeve, numNeck,
        finalAct="softmax"):
        # initialize the input shape and channel dimension
        inputShape = (height, width, 3)
        chanDim = -1
        # construct the pattern, sleeve and neck sub-networks
        inputs = Input(shape=inputShape)
        patternBranch = Flix.build_pattern_branch(inputs,
            numCategories = numPattern, finalAct = finalAct, chanDim=chanDim)
        sleeveBranch = Flix.build_sleeve_branch(inputs,
            numCategories=numSleeve, finalAct=finalAct, chanDim=chanDim)
        neckBranch = Flix.build_neck_branch(inputs,
            numCategories=numNeck, finalAct=finalAct, chanDim=chanDim)
        # create the model using our input (the batch of images) and
        
        model = Model(
            inputs=inputs,
            outputs=[neckBranch,sleeveBranch,patternBranch],
            name="FlixStock")
        # return the constructed network architecture
        return model