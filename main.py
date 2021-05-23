############################### #how to read an image
# import cv2
# img=cv2.imread('1.jpg')

# cv2.imshow('output image',img)

# cv2.waitKey(0)

# # cv2.destroyAllWindows()


########################how to write an image ?
# import cv2
# img=cv2.imread('1.jpg')


# cv2.imshow('Orginal Image',img)
# cv2.imwrite('output.jpg',img)
# cv2.imwrite('output.png',img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#######################how image info?

# import cv2
# img=cv2.imread('1.jpg')


# cv2.imshow('Orginal Image',img)
# print(img.shape)
# print("heiht Pixel value:",img.shape[0])
# print("weigth Pixel value:",img.shape[1])

# cv2.waitKey(0)
# cv2.destroyAllWindows()



#######################how to convert RGB to gray scale image?
# import cv2
# img=cv2.imread("1.jpg")
# cv2.imshow('orginal',img)
# cv2.waitKey(0)
# gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.destroyAllWindows()
# cv2.imshow("Gray scale Imag",gray_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#######################how to convert RGB to Binary img?
# import cv2
# img=cv2.imread("3.png",0)
# cv2.imshow("Gray",img)

# cv2.waitKey(0)


# ret,bw=cv2.threshold(img,100,200,cv2.THRESH_BINARY)
# cv2.imshow("Binary",bw)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#######################how to convert RGB to Binary img?
# import cv2
# img=cv2.imread("3.png")
# img_HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# cv2.imshow('HSV Image',img_HSV)
# cv2.imshow('Hue channel',img_HSV[:,:,0])
# cv2.imshow('saturation',img_HSV[:,:,1])
# cv2.imshow('Value channel',img_HSV[:,:,2])

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#######################how to convert RGB to Binary img?
# import cv2
# import numpy as np

# img=cv2.imread('3.png')
# cv2.imshow("orginal",img)
# cv2.waitKey(0)
# B,G,R=cv2.split(img)
# zeros= np.zeros(img.shape[:2], dtype="uint8")

# cv2.imshow("Red",cv2.merge([zeros,zeros,R]))
# cv2.waitKey(0)

# cv2.imshow("Green",cv2.merge([zeros,G,zeros]))
# cv2.waitKey(0)

# cv2.imshow("Blue",cv2.merge([B,zeros,zeros]))
# cv2.waitKey(0)

# cv2.destroyAllWindows()

############# image translation
# import cv2
# import numpy as np

# img=cv2.imread('3.png')

# height,width=img.shape[:2]
# print(height)
# print(width)
# quarter_height,quarter_width=height/4,width/4
# print(quarter_height)
# print(quarter_width)

# #  T=|1 0 Tx|
# #    |0 1 Ty|
# T= np.float32([[1,0,quarter_width],
#                 [0,1,quarter_height]])
# print(T)
# #we use worpaffine transformation to shift the image
# img_translation=cv2.warpAffine(img,T,(width,height))

# cv2.imshow('orginal Image',img)
# cv2.imshow('Translation',img_translation)
# cv2.waitKey(0)
# cv2.destroyallwindows


############# image rotation
# import cv2
# import numpy as np

# img=cv2.imread('3.png')
# height,width=img.shape[:2]
# print(height)
# print(width)
# rotation_matrix=cv2.getRotationMatrix2D((width/2,height/2),-20,.5)
# rotated_image=cv2.warpAffine(img,rotation_matrix,(width,height))
# cv2.imshow('Rotated image',rotated_image)
# cv2.imshow('orginal Image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

############# image resize and interplotation
import cv2
import numpy as np
img=cv2.imread('3.png')
#lets make the size of our image 3/4 of its 
img_scaled=cv2.resize(img,None,fx=0.75,fy=0.75)
cv2.imshow('orginal Image',img)
cv2.imshow('sclaing-linear Interplotion',img_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()

#lets make the double the size of orginal image
img_scaled=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
cv2.imshow('sclaing-cubic Interplotion',img_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()

#lets skew the resizing by setting extact dimension
img_scaled=cv2.resize(img,(900,400),interpolation=cv2.INTER_AREA)
cv2.imshow('sclaing-skewes sized',img_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()