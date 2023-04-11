import matplotlib.pyplot as plt
import sys
import os
import h5py as hp
import numpy as np
import scipy.ndimage as nd
import cv2
import matplotlib.patches as patch



file = sys.argv[1]
prefix = file[:-3]
print(prefix)
data = hp.File(file,'r')
image_raw = np.array(data.get('pds_image'))
image_raw = image_raw[91:,80:]


# plot filtered image:
plt.figure()
image_filt = nd.median_filter(image_raw, 5)
image_filt = nd.gaussian_filter(image_filt, 5)
plt.imshow(image_filt)
plt.colorbar()

#Plot the original image:
#plt.figure()
#plt.imshow(image_raw, cmap='bone')
#plt.colorbar()

# find the centers of all of the images (user input):
print('Choose centers of circles (as close as you can):')
clicks = plt.ginput(16)
print(clicks)


#making some placeholder lists for images, image indices, binary images and raw images (cropped around each circle in the image):
images_filter = []
image_num = []
images_raw = []


width = 170
image_count = 0
size1,size2 = image_filt.shape

#cropping out individual circles: 
for click in clicks:
	cl1 = int(click[1])
	cl2 = int(click[0])

	print(cl1)
	print(cl2)
	image_count += 1
	
	width11 = min([width, cl1-1])
	width12 = min([width, size1 - cl1 - 1])

	width21 = min([width, cl2-1])
	width22 = min([width, size2-cl2-1])
	
	try:
		
		image_raw_crop = image_raw[cl1-width11:cl1+width12, cl2-width21:cl2+width22]   
		image_filt_crop = image_filt[cl1-width11:cl1+width12, cl2-width21:cl2+width22]   
		#image_bin_crop = image_bin[cl1-width11:cl1+width12, cl2-width21:cl2+width22]	
		#images_bin.append(image_bin_crop)
		#images_bin_filter.append(image_bin_filter[cl1-width11:cl1+width12, cl2-width21:cl2+width22])
		images_raw.append(image_raw_crop)
		images_filter.append(image_filt_crop)
		image_num.append(image_count)
	except(IndexError):
		pass

#scaling all of the images between max and min: 

images_bin = []
for image_crop in images_filter:
	plt.figure()
	plt.imshow(image_crop)

	maxmin = plt.ginput(2)
	max_val = maxmin[0]
	max_val_1 =int(max_val[1])
	max_val_2 = int(max_val[0])

	min_val = maxmin[1]
	min_val_1 = int(min_val[1])
	min_val_2 = int(min_val[0])

	min_scale = np.average(image_crop[min_val_1-10:min_val_1+10, min_val_2-10:min_val_2+10])

	max_scale = np.average(image_crop[max_val_1-10:max_val_1+10, max_val_2-10:max_val_2+10])

	image_scaled = (image_crop - min_scale)/(max_scale - min_scale)
	
	image_bin_filter = image_scaled * (image_scaled > .5)
	image_bin = (image_scaled > .5)
	#plt.figure()
	#plt.imshow(image_bin_filter)
	images_bin.append(image_bin)
	plt.close()	


fig, ax = plt.subplots(4,4)
for x in range(4):
	for y in range(4):
		try:
			im_bin = images_bin[4*x + y]
			ax[x,y].imshow(im_bin)
		except(IndexError):
			pass


fig, ax = plt.subplots(4,4)
for x in range(4):
	for y in range(4):
		try:
			im = images_raw[4*x + y]
			ax[x,y].imshow(im)
		except(IndexError):
			pass



circle_analysis  = True

if circle_analysis:

	im_edges =[]
	rs = [] # all of the radii that we collect
	image_indices = []

	for x in range(4):
		for y in range(4):
			try:
				im_bin = images_bin[4*x + y]
				im_bin = np.array(im_bin, dtype = np.uint8) 
				im_norm = cv2.normalize(src=im_bin, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
				# Apply Hough transform on the blurred image
				detected_circles = cv2.HoughCircles(im_norm, 
						   cv2.HOUGH_GRADIENT, 1, minDist = 500, param1 = 25,
					       param2 = 5, minRadius = 30, maxRadius = 200)
				
				try:	
					for circle in detected_circles[0]:
						a,b,r = circle[0], circle[1], circle[2]
						rs.append(r)
						image_indices.append(4*x + y)
						circ = patch.Circle((a,b), fill=False, edgecolor = 'r', linewidth = 1, radius = r)
						ax[x,y].add_patch(circ)
				except(TypeError):
					pass
			except(IndexError):
				pass
	
plot_rs = True
if plot_rs:
	

	plt.figure()
	plt.scatter(image_indices, rs)
	print('select points you would like to exclude')
	exclude_points = plt.ginput(16)

	#exclude_input = input('enter all of the frame numbers that you would like to exclude (index from 0), separated by commas: ' )
	exclude_list = []
	for exclude_point in exclude_points:
		exclude_list.append(round(exclude_point[0]))

	print(exclude_list) 

	for image_ind in  exclude_list:
		try:
			list_ind = image_indices.index(image_ind)
			image_indices.remove(image_ind)
			rs.remove(rs[list_ind])
		except(ValueError):
			pass

	plt.figure()
	plt.scatter(image_indices, rs)
	plt.title('inlier data points')

	with open(prefix + '.txt', 'w') as save_file:
		for element in range(len(image_indices)):
			save_file.write(str(image_indices[element]) + ', ' + str(rs[element]) + '\n')


			
plt.show()



