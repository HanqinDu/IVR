import cv2
import numpy as np


# This code uses support vector machine on the flat part images
# There are three classes; part1, part2, and part3
# These training samples are stored in the folder train_images
# There s a test image in the directory test_image which is part2
#
# This code is under the assumption that the images are black and
# white

class classifier():

	# classes should be a 2D array where
	# the first value is the name of the class,
	# and the second is the number of training
	# instances
	def __init__(self, classes, no_dim=3):
		self.svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR,
                    svm_type = cv2.ml.SVM_C_SVC,
                    C=2.67, gamma=5.383 )
		self.classes=classes
		self.dim=no_dim

		self.samples=np.array([[]], dtype=np.float32)
		self.samples_labels=np.array([], dtype=np.int)
		self.svm = cv2.ml.SVM_create()
		self.svm.setType(cv2.ml.SVM_C_SVC)
		self.svm.setKernel(cv2.ml.SVM_LINEAR)
		self.svm.setTermCriteria((cv2.	TERM_CRITERIA_COUNT, 100, 1.e-06))

	#######################from lec
	def loadImage(self,filename):
		img = cv2.imread(filename)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return img_gray

	def peakPick(self, img):
	    img_hist=cv2.calcHist([img],[0],None,[256],[0,256])
	    img_hist= self.threshHistSmooth(img_hist, 50, 5, 1)

	    # Find peak in histogram
	    peak = np.argmax(img_hist)
	    # Find peak in the darker side of the above peak value
	    peak_darker = 1
	    for i in range(1,peak):
	    	if img_hist[i-1] < img_hist[i] and img_hist[i] >= img_hist[i+1] and img_hist[i]>img_hist[peak_darker]:
	    		peak_darker=i
	    # print peak_darker
	    # Find the deepest valley bewteen the two peaks
	    # This will be our threshold value
	    thresh = peak_darker+1
	    for i in range(peak_darker+1,peak):
	    	if img_hist[i-1] > img_hist[i] and img_hist[i] <= img_hist[i+1] and img_hist[i] < img_hist[thresh]:
	    		thresh = i
	    return thresh

	def threshold(self, img, thresh_value):
		img = np.array(img)
		height,width = np.shape(img)
		output=np.zeros([height,width])
		for row in range(height):
			for col in range(width):
				if img[row][col]>thresh_value:
					output[row][col] = 1

		return output

	def threshHistSmooth(self, img_hist, n, w, sigma):
	    nn = int((n-1)/2)
	    gauss = np.asarray([(w/n)*x**2 for x in range(-nn,nn+1)], dtype=float)
	    gauss = np.exp(-gauss/(2*sigma**2))
	    the_filter = gauss/sum(gauss)
	    hist_convolve = np.convolve(np.ravel(img_hist), the_filter)
	    return hist_convolve

	def bwperim(self,img):
		img=np.array(img)
		height,width = np.shape(img)
		perim_img=np.zeros([height,width])
		for row in range(height):
			for col in range(width):
				if row==0 or row==height-1 or col==0 or col==width-1:
					if img[row][col]==1:
						perim_img[row][col]=1
				else:
					if img[row][col]==1 and (img[row][col+1]==0 or img[row+1][col+1]==0 or img[row+1][col]==0 or img[row+1][col-1]==0 or img[row][col-1]==0 or img[row-1][col-1]==0 or img[row-1][col]==0 or img[row-1][col+1]==0):
						perim_img[row][col]=1
		# Uncomment the next two lines if you want to see the perimeter
		# cv2.imshow("perim image",perim_img)
		# cv2.waitKey(5)
		return sum(perim_img.ravel())
	
	def complexmoment(self, img, u, v):
		img = np.array(img)
		indices = np.argwhere(img>0)
		centre = np.mean(indices,0)

		momlist = np.zeros(np.shape(indices)[0],dtype=complex)
		for i in indices:
			c1 = i[0]-centre[0]+(i[1]-centre[1])*1j
			c2 = i[0]-centre[0]+(centre[1]-i[1])*1j
			momlist[i]=np.power(c1,u)*np.power(c2,v)

		return sum(momlist)

	def getproperties(self, img):
		img = np.array(img)
		area = sum(img.ravel())

		perim = self.bwperim(img)

		compactness = perim*perim/(4.0*np.pi*area)

		c11 = self.complexmoment(img,1,1) / np.power(area,2)
		c20 = self.complexmoment(img,2,0) / np.power(area,2)
		c30 = self.complexmoment(img,3,0) / np.power(area,2.5)
		c21 = self.complexmoment(img,2,1) / np.power(area,2.5)
		c12 = self.complexmoment(img,1,2) / np.power(area,2.5)

		ci1 = np.real(c11)
		ci2 = np.real(1000*c21*c12)
		tmp = c20*c12*c12
		ci3 = 10000*np.real(tmp)
		ci4 = 10000*np.imag(tmp)
		tmp = c30*c12*c12*c12
		ci5 = 1000000*np.real(tmp)
		ci6 = 1000000*np.imag(tmp)

		return [compactness, ci1, ci2]


	def threshold(self, img, thresh_value):
		img = np.array(img)
		height,width = np.shape(img)
		output=np.zeros([height,width])
		for row in range(height):
			for col in range(width):
				if img[row][col]>thresh_value:
					output[row][col] = 1
		return output

	#####################################

	def addTrainSamples(self, folder):
		kernel = np.ones((5,5),np.uint8)
		class_key=0
		for c in self.classes:
			for i in range(1,c[1]+1):
				filename=folder+'/'+c[0]+str(i)+'.jpg'
				img=self.loadImage(filename)
				thresh=self.peakPick(img)
				img_bin=abs(self.threshold(img, thresh)-1)
				# Given this set of data we need to flip the values
				img_bin=cv2.erode(img_bin,kernel,iterations=2)
				img_props=np.array([self.getproperties(img_bin)], dtype=np.float32)
				if self.samples.size==0:
					self.samples=img_props
				else:
					self.samples=np.append(self.samples,img_props, axis=0)
				self.samples_labels=np.append(self.samples_labels, np.array([class_key], dtype=np.int))
			class_key+=1
		return

	def train(self, model="", load=False):
		if load:
			try:
				self.svm = self.svm.load(model)
			except:
				print "Provide a valid xml file."
		else:
			self.svm.train(self.samples, cv2.ml.ROW_SAMPLE, self.samples_labels)
			if model!="":
				try:
					self.svm.save(model)
				except:
					print "The filename must be valid."
		return

	def classify(self, filename):
		img=self.loadImage(filename)
		kernel = np.ones((5,5),np.uint8)
		thresh=self.peakPick(img)
		img_bin=abs(self.threshold(img, thresh)-1) # Given this set of data we need to flip the values
		img_bin=cv2.erode(img_bin,kernel,iterations=2)
		img_props=np.array([self.getproperties(img_bin)], dtype=np.float32)
		prediction=self.svm.predict(img_props)
		return prediction

def main():

	classes=[['invalidxy', 156],['validxy', 154]]

	test_classifier=classifier(classes)

	## APPROACH 1 ##
	# The following code can be used if there is not
	# a model to be loaded and must create one from
	# scratch.
	test_classifier.addTrainSamples('trainData')
	test_classifier.train()

	return test_classifier
