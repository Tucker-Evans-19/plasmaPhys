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
plt.figure()

#image_bg = image_raw - np.average(image_raw[100:200, 50:150])



#image_bg = image_raw/np.max(image_raw)


image_filt = nd.median_filter(image_raw, 15)
#image_filt = nd.gaussian_filter(image_filt, 20)
#image_filt = (image_filt > .2*np.max(image_filt))

plt.imshow(image_filt)
plt.colorbar()
print('select peak and valley locations')

maxmin = plt.ginput(2)
max_val = maxmin[0]
max_val_1 =int(max_val[1])
max_val_2 = int(max_val[0])

min_val = maxmin[1]
min_val_1 = int(min_val[1])
min_val_2 = int(min_val[0])

min_scale = np.average(image_filt[min_val_1-10:min_val_1+10, min_val_2-10:min_val_2+10])

max_scale = np.average(image_filt[max_val_1-10:max_val_1+10, max_val_2-10:max_val_2+10])

image_scaled = (image_filt - min_scale)/(max_scale - min_scale)
plt.figure()

image_bin_filter = image_scaled * (image_scaled > .1)
image_bin = (image_scaled > .2)

plt.imshow(image_bin, cmap='bone')
plt.colorbar()

clicks = plt.ginput(16)

#clicks = [(1627.5241935483868, 136.43951612903243), (1191.5362903225805, 154.08064516129048), (735.3870967741933, 141.47983870967755), (299.3991935483871, 121.31854838709683), (1632.5645161290317, 532.1048387096776), (1176.415322580645, 542.1854838709678), (745.4677419354838, 542.1854838709678), (306.95967741935476, 562.3467741935485), (1645.165322580645, 963.0524193548385), (1211.697580645161, 968.0927419354839), (758.0685483870967, 975.6532258064515), (327.1209677419354, 993.2943548387095), (1665.3266129032254, 1404.0806451612902), (1224.2983870967737, 1409.1209677419354), (778.2298387096772, 1424.2419354838707), (334.6814516129032, 1439.3629032258063)]


print(clicks)
#plt.figure() 
#plt.plot(clicks)

images_bin_filter = []
image_num = []
images_bin = []


width = 170
image_count = 0
size1,size2 = image_bin.shape

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
		
			
		image_bin_crop = image_bin[cl1-width11:cl1+width12, cl2-width21:cl2+width22]	
		images_bin.append(image_bin_crop)
		images_bin_filter.append(image_bin_filter[cl1-width11:cl1+width12, cl2-width21:cl2+width22])

		image_num.append(image_count)
	except(IndexError):
		pass
	
#fig, ax = plt.subplots(4,4)
im_edges =[]

fig1, ax1 = plt.subplots(4,4)
#fig2, ax2 = plt.subplots(4,4)
rs = [] # all of the radii that we collect
image_indices = []

for x in range(4):
	for y in range(4):
#		ax[int(np.floor(image_num[el]/4)), image_num[el]%4].imshow(images[el])
		try:
			im_bin = images_bin[4*x + y]
			im_bin_filter = images_bin_filter[4*x + y]
			
			im_bin = np.array(im_bin, dtype = np.uint8) 
			im_norm = cv2.normalize(src=im_bin, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
			# Blur using 3 * 3 kernel.
			#gray_blurred = cv2.blur(gray, (5, 5))
			ax1[x,y].imshow(im_bin_filter)
			# Apply Hough transform on the blurred image.
			detected_circles = cv2.HoughCircles(im_norm, 
					   cv2.HOUGH_GRADIENT, 1, minDist = 500, param1 = 25,
				       param2 = 8, minRadius = 30, maxRadius = 200)
			#print('These are the circles detected: \n')			
			#print(detected_circles)
			try:	
				for circle in detected_circles[0]:
					a,b,r = circle[0], circle[1], circle[2]
					rs.append(r)
					image_indices.append(4*x + y)
					#print('this is a:')
					#print(a)
					#print('this is b:')
					#print(b)
					#print('this is r:')
					#print(r)
					#print(circle)
					circ = patch.Circle((a,b), fill=False, edgecolor = 'r', linewidth = 3, radius = r)
					ax = ax1[x,y]
					ax.add_patch(circ)
			#	ax1[x,y].imshow(im)
			except(TypeError):
				pass
		except(IndexError):
			pass

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



