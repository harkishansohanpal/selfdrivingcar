import argparse
import base64
from datetime import datetime
import json
import shutil
import cv2
import numpy as np
import time
from moviepy.editor import VideoFileClip
from PIL import Image
from PIL import ImageOps
import os
import numpy as np
from config import *
from load_data import preprocess
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from scipy.misc import imresize
from keras.models import load_model
# Fix error with Keras and TensorFlow
import tensorflow as tf
model1 = load_model('full_CNN_model.h5')
model = None
prev_image_array = None

class Lanes():
	def __init__(self):
		self.recent_fit = []
		self.avg_fit = []
def road_lines(image):
	""" Takes in a road image, re-sizes for the model,
	predicts the lane to be drawn from the model in G color,
	recreates an RGB image of a lane and merges with the
	original road image."""
	
	

	# Get image ready for feeding into model
	small_img = imresize(image, (80, 160, 3))
	small_img = np.array(small_img)
	small_img = small_img[None,:,:,:]

	# Make prediction with neural network (un-normalize value by multiplying by 255)
	prediction = model1.predict(small_img)[0] * 255

	# Add lane prediction to list for averaging
	lanes.recent_fit.append(prediction)
	# Only using last five for average
	if len(lanes.recent_fit) > 5:
		lanes.recent_fit = lanes.recent_fit[1:]

	# Calculate average detection
	lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

	# Generate fake R & B color dimensions, stack with G
	blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
	lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

	# Re-size to match the original image
	lane_image = imresize(lane_drawn, (720, 1280, 3))

	# Merge the lane drawing onto the original image
	result = cv2.addWeighted(image, 1, lane_image, 1, 0)

	return result
lanes = Lanes()


def telemetry():

	cap = cv2.VideoCapture('data/project_video.mp4')

	while(cap.isOpened()):
		ret, frame = cap.read()
		
		image_array = cv2.cvtColor(np.asarray(frame), code=cv2.COLOR_RGB2BGR)
		# perform preprocessing (crop, resize etc.)
		image_array = preprocess(frame_bgr=image_array)

		# add singleton batch dimension
		image_array = np.expand_dims(image_array, axis=0)

		# This model currently assumes that the features of the model are just the images. Feel free to change this.
		steering_angle = float(model.predict(image_array, batch_size=1))+0.15

		# The driving model currently just outputs a constant throttle. Feel free to edit this.
		throttle = 0.28
		print(steering_angle, throttle)
		frame=road_lines(frame)
		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		if args.image_folder != '':
			timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
			image_filename = os.path.join(args.image_folder, timestamp)
			cv2.imwrite('{}.jpg'.format(image_filename),frame)
	cap.release()
	cv2.destroyAllWindows()
	

if __name__ == '__main__':

	from keras.models import model_from_json

	parser = argparse.ArgumentParser(description='Remote Driving')
	
	parser.add_argument(
		'image_folder',
		type=str,
		nargs='?',
		default='',
		help='Path to image folder. This is where the images from the run will be saved.'
	)
	args = parser.parse_args()

	json_path ='pretrained/model.json'
	with open(json_path) as jfile:
		model = model_from_json(jfile.read())
	# load model weights
	# weights_path = os.path.join('checkpoints', os.listdir('checkpoints')[-1])
	weights_path = 'pretrained/model.hdf5'
	print('Loading weights: {}'.format(weights_path))
	model.load_weights(weights_path)
	if args.image_folder != '':
		print("Creating image folder at {}".format(args.image_folder))
		if not os.path.exists(args.image_folder):
			os.makedirs(args.image_folder)
		else:
			shutil.rmtree(args.image_folder)
			os.makedirs(args.image_folder)
		print("RECORDING THIS RUN ...")
	else:
		print("NOT RECORDING THIS RUN ...")
	# compile the model
	model.compile("adam", "mse")

	telemetry()
