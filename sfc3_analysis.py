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

plt.figure()

image_bg = image_raw - np.average(image_raw[100:200, 50:150])
image_filt = nd.median_filter(image_bg, 5)
image_filt = nd.gaussian_filter(image_filt, 5)


plt.imshow(image_filt)
plt.colorbar()



clicks = plt.ginput(16)
print(clicks)
plt.figure() 
plt.plot(clicks)

images = []
image_num = []

width = 200
image_count = 0

for click in clicks:
	cl1 = int(click[1])
	cl2 = int(click[0])

	print(cl1)
	print(cl2)
	image_count += 1
	try:
		image = image_filt[cl1-width:cl1+width, cl2-width:cl2+width]	
		images.append(image)
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
			im = images[4*x + y]
			#ax1[x,y].imshow(im)
			#im_blur = cv2.GaussianBlur(im, (3,3), 0)
			#im_edge = cv2.Sobel(src = im_blur, ddepth=cv2.CV_64F, dx =1,dy=1,ksize=5)		
			#im_edges.append(im_edge)
			#plt.figure()
			#plt.pcolor((im > .5*np.max(im)))

			#im_edge = cv2.Canny(im, threshold1 = 100, threshold2 = 200)
			#ax2[x,y].pcolor(np.abs(im_edge))
			#ax2[x,y].colorbar()
			#plt.figure()
			#im_edge_cutoff =(im_edge > 40)
			#plt.pcolor(np.abs(im_edge))
			#plt.colorbar()
			#plt.figure()
			#plt.pcolor(im * (im >.75*np.max(im)))

			# trying Hough Circles technique:
			# Convert to grayscale.
			#gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			gray = np.array(im)/np.max(im) 

			gray = cv2.normalize(src=gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
			# Blur using 3 * 3 kernel.
			gray_blurred = cv2.blur(gray, (5, 5))
				  
			ax1[x,y].imshow(im)
			# Apply Hough transform on the blurred image.
			detected_circles = cv2.HoughCircles(gray_blurred, 
					   cv2.HOUGH_GRADIENT, 1, minDist = 50, param1 = 50,
				       param2 = 30, minRadius = 1, maxRadius = 200)
			
			print(detected_circles)
			try:	
				for circle in detected_circles[0]:
					a,b,r = circle[0], circle[1], circle[2]
					rs.append(r)
					image_indices.append(4*x + y)
					print('this is a:')
					print(a)
					print('this is b:')
					print(b)
					print('this is r:')
					print(r)
					print(circle)
					circ = patch.Circle((a,b), radius = r)
					ax = ax1[x,y]
					ax.add_patch(circ)
			#	ax1[x,y].imshow(im)
			except(TypeError):
				pass
		except(IndexError):
			pass

plt.figure()
plt.scatter(image_indices, rs)
print(rs)
		
with open(prefix + '.txt', 'w') as save_file:
	for element in range(len(image_indices)):
		save_file.write(str(image_indices[element]) + ', ' + str(rs[element]) + '\n')


		
plt.show()




