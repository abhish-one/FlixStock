# FlixStock-assignment

This repository contains the code for Classification of Men-Tshirt's Attribues.

Basic Requirement : Python 3.7, Tensorflow2.0

Project Structure :  It contains two seperate directories : 1. data - it contains images, attribute.csv and test directory.
                                                            2. output - this directory hold the output models.
                                                            

## Steps to train and predict : 
1. Download the zip or clone this repository
2. Run train.py : python train.py --dataset data --model output/flixstock.model /
                                        --patternbin output/pattern_lb.pickle --neckbin output/neck_lb.pickle --sleevebin output/sleeve_lb.pickle
3. Run classify.py : python classify.py --model output/flixstock.model --patternbin output/pattern_lb.pickle /
                                        --neckbin output/neck_lb.pickle --sleevebin output/sleeve_lb.pickle --image data/test/test1.jpg 
                                        
 On running classify.py for a given image it will output an image with predicted labels along with their probabilities.
 
 
 Note : I have trained a Multi-output Neural network to predict each attribute. Hence used the keras functional api for the same.
 
 I have used custom three layer Convolutional model to train this dataset, however I also tried to train using pre-trained Vgg19 but wasn't able to 
 succeed in the given time frame.
 
The code is modular enough to change Convolutional Network architecture very easily and train again. 

** For further improvement we can also first try to detect the t-shirt in a full image and then try to do classification.
