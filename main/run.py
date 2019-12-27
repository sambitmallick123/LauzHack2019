#imported libs
#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import os
import sys
import json
import datetime
import skimage.draw
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img,img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances
from skimage.restoration import inpaint
from skimage import data
import tweepy
from textblob import TextBlob
import pyttsx3
import string
from gtts import gTTS
import moviepy.editor as mpe
#from wordcloud import WordCloud, STOPWORDS


############################################################################################


""" Description


"""

""" AUXILIARY FUNCTION """

alpha_num = string.digits+string.ascii_letters
alpha_num = [char for char in alpha_num]
engine = pyttsx3.init()

class TestConfig(Config):
	 NAME = "ad"
	 GPU_COUNT = 1
	 IMAGES_PER_GPU = 1
	 NUM_CLASSES = 1 + 1

def whiten(img,mask_array):
	for i in range(mask_array.shape[0]):
		for j in range(mask_array.shape[1]):
			if 1 in mask_array[i,j]:
				img[i,j,0] = 255
				img[i,j,1] = 255
				img[i,j,2] = 255
				mask_array[i,j,0] = 1
				mask_array[i,j,1] = 1
				mask_array[i,j,2] = 1
	return img,mask_array

def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < 0.5:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

def convert_frames_to_video(frame_array,pathOut,fps):
	#frame_array already ordered
	#for sorting the file names properly

	height, width, layers = frame_array[0].shape
	size = (width,height)
	
	try:
		out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
	except:
		print('no video written')

	for i in range(len(frame_array)):
		
		# writing to a image array
		out.write(frame_array[i])
	out.release()

def most_frequent(List): 
	return max(set(List), key = List.count) 

############################################################################################

""" MAIN FUNCTIONS """


def image(image_path,pathout='result.jpg'):
	pass


def video_out(video_path,pathout='result.mp4',fps=30):
	#Globals
	cap = cv2.VideoCapture(video_path)
	array_Images = []
	
	#Settings for Mask
	rcnn = MaskRCNN(mode='inference', model_dir='./load_weights', config=TestConfig())
	rcnn.load_weights('../weights/mask_rcnn_ad.h5', by_name=True)


	while True:
		ret,image = cap.read()
		if not ret: break
		#Sementing and then whitening out
		results = rcnn.detect([image], verbose=0)
		mask_array = results[0]['masks']
		image1 ,mask_array = whiten(image,mask_array)

		#Inpainting process
		mask_array = mask_array[:,:,1].reshape((mask_array.shape[0],mask_array.shape[1]))
		image_result = inpaint.inpaint_biharmonic(image1,mask_array,multichannel=True)
		array_Images.append(image_result)

	# LAST PART 
	#Merging pic for a video
	convert_frames_to_video(array_Images,pathout,fps)

def audio_out(video_path,pathout='mixed_result.mp4',fps=30): #outputing audio

	cap = cv2.VideoCapture(video_path)
	

	#Settings for Text
	(newW, newH) = (320, 320)
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]
	print("[INFO] loading EAST text detector...")
	net = cv2.dnn.readNet('../weights/frozen_east_text_detection.pb')

	sampling_rate = 100

	#Each interval of sometimes we fetch call this feature here

	results_texts = []
	while True:
		frameId = cap.get(1)
		ret,image = cap.read()
		if not ret: break

		if not (frameId % 100) :
			orig = image.copy()
			(origH, origW) = image.shape[:2]
			rW, rH = origW / float(newW) , origH / float(newH)
			image = cv2.resize(image, (newW, newH))
			(H, W) = image.shape[:2]
			blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
				(123.68, 116.78, 103.94), swapRB=True, crop=False)
			net.setInput(blob)
			(scores, geometry) = net.forward(layerNames)

			# decode the predictions, then  apply non-maxima suppression to
			# suppress weak, overlapping bounding boxes
			(rects, confidences) = decode_predictions(scores, geometry)
			boxes = non_max_suppression(np.array(rects), probs=confidences)

			# initialize the list of results
			texts = []

			# loop over the bounding boxes
			for (startX, startY, endX, endY) in boxes:
				# scale the bounding box coordinates based on the respective
				# ratios
				startX = int(startX * rW)
				startY = int(startY * rH)
				endX = int(endX * rW)
				endY = int(endY * rH)

				# in order to obtain a better OCR of the text we can potentially
				# apply a bit of padding surrounding the bounding box -- here we
				# are computing the deltas in both the x and y directions
				pad = 0
				dX = int((endX - startX) * pad)
				dY = int((endY - startY) * pad)

				# apply padding to each side of the bounding box, respectively
				startX = max(0, startX - dX)
				startY = max(0, startY - dY)
				endX = min(origW, endX + (dX * 2))
				endY = min(origH, endY + (dY * 2))

				# extract the actual padded ROI
				roi = orig[startY:endY, startX:endX]
				config = ("-l eng --oem 1 --psm 7")
				text = pytesseract.image_to_string(roi, config=config)
				nice_text = ""
				for char in text: 
					if char in alpha_num: nice_text = nice_text + char
				if len(nice_text) != 0: texts.append(nice_text)
			results_texts.append(texts)	

	lists = []
	for array in results_texts:
		for element in array: lists.append(element)

	#print(lists)
	string = most_frequent(lists)

	consumer_key= "9F9eMYUbCF4IHMY4pKpJCTwio"
	consumer_secret= "Kb77J6szyBVhmeA3hwk7ubOSi1Acn54kXxh7V7O12K99ThP1xa"
	access_token="90417751-3Jy9wwyiSc85lZkYRG0LNowQUTztrJ26HiNvidEW9"
	access_token_secret="oQ7v9UcLmpmW1HDLLNaxCtc5xpqw23CY3FhcvsqFCRnOy"
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)

	try: 
		public_tweets = api.search(string)
		
		for tweet in public_tweets:

			twtText=tweet.text.encode('unicode-escape').decode('UTF-8')
	
			with open("test.txt", "a") as myfile: myfile.write(twtText)
			print(tweet.text.encode('unicode-escape').decode('UTF-8'))
			analysis = TextBlob(tweet.text)
			print(analysis.sentiment)
	except:
		print('word not found by the api on twitter or other errors')
	
	infile = "test.txt"
	with open(infile, 'r') as f: theText = f.read()

	#Saving part starts from here 
	tts = gTTS(text=theText, lang='en')
	tts.save("saved_file.mp3")
	print("File saved!")

	my_clip = mpe.VideoFileClip(video_path)
	audio_background = mpe.AudioFileClip('saved_file.mp3')
	final_audio = mpe.CompositeAudioClip([my_clip.audio, audio_background])
	final_clip = my_clip.set_audio(final_audio)

############################################################################################



if __name__ == '__main__':
	array = sys.argv[1:]
	if len(array) != 2: sys.exit(0)
	if not os.path.exists(array[1]): sys.exit(0)
	if array[0] == '-v': video_out(array[1])
	if array[0] == '-a': audio_out(array[1])
	else: sys.exit(0)

