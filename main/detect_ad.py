import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img,img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances

""" Description


"""


######################################################################################################


#classes --> just ignore the first class, I'll sort that out some other time
class_names = ['.','ad'] 
 
# define the test configuration
class TestConfig(Config):
	 NAME = "ad"
	 GPU_COUNT = 1
	 IMAGES_PER_GPU = 1
	 NUM_CLASSES = 1 + 1


#########################################################################################

def whiting_segmented(img,mask_results):
	#img is in already array size
	#mask_results is the dictionary outputed from the mask model

	mask_array = mask_results['masks'] #we change the mask to do simple multiplication after that	
	
	for i in range(mask_array.shape[0]):
		for j in range(mask_array.shape[1]):
			if 1 in mask_array[i,j]:
				img[i,j,0] = 255
				img[i,j,1] = 255
				img[i,j,2] = 255
				mask_array[i,j,0] = 1
				mask_array[i,j,1] = 1
				mask_array[i,j,2] = 1
	np.save('assets/mask_example.npy',mask_array)
	cv2.imwrite('assets/whitenedAd.jpg',img)
 	

######################################################################################################


def main(path_to_image,path_to_weight='../weights/mask_rcnn_ad.h5'):
	# define the model
	rcnn = MaskRCNN(mode='inference', model_dir='./load_weights', config=TestConfig())
	# load coco model weights
	rcnn.load_weights(path_to_weight, by_name=True)
	
	# load photograph
	img = load_img(path_to_image)
	img = img_to_array(img)
	
	# make prediction
	results = rcnn.detect([img], verbose=0)
	# get dictionary for first prediction
	r = results[0]
	img = cv2.imread(path_to_image) #Will fix it another time, it's working fine just ugly
	# show photo with bounding boxes, masks, class labels and scores
	#display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
	#saving pic and mask matrix into asset
	whiting_segmented(img,r)

if __name__ == '__main__':

	array = sys.argv[1:]
	if len(array) != 1: 
		print('Wrong Args')
		sys.exit(0)
	elif not os.path.exists(array[0]): 
		print('Path to image not correct')
		sys.exit(0)

	main(array[0])

