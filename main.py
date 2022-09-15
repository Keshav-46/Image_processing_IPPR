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
# import cv2
# import numpy as np
# img=cv2.imread('3.png')
# #lets make the size of our image 3/4 of its 
# img_scaled=cv2.resize(img,None,fx=0.75,fy=0.75)
# cv2.imshow('orginal Image',img)
# cv2.imshow('sclaing-linear Interplotion',img_scaled)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #lets make the double the size of orginal image
# img_scaled=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
# cv2.imshow('sclaing-cubic Interplotion',img_scaled)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #lets skew the resizing by setting extact dimension
# img_scaled=cv2.resize(img,(900,400),interpolation=cv2.INTER_AREA)
# cv2.imshow('sclaing-skewes sized',img_scaled)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##############   image resize using image pyramid
# import cv2
# import numpy as pd

# img=cv2.imread('3.png')
# smaller=cv2.pyrDown(img)
# larger=cv2.pyrUp(img)

# cv2.imshow('Orginal',img)
# cv2.imshow('smaller',smaller)
# cv2.imshow('Larger',larger)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##############   image cropping
# import cv2
# import numpy as pd
# img=cv2.imread('3.png')

# height,width=img.shape[:2]

# start_row,start_col=int(height*0.25),int(height*0.25)
# end_row,end_col=int(height*0.75),int(height*0.75)

# cropped=img[start_row:end_row,start_col:end_col]

# cv2.imshow('orginal image',img)
# cv2.waitKey(0)
# cv2.imshow('cropped',cropped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##############   image Bitwise operation
# import cv2
# import numpy as np
# square=np.zeros((300,300),np.uint8)
# cv2.rectangle(square,(50,50),(250,250),255,3)
# cv2.imshow("Square",square)
# cv2.waitKey(0)

# ellipse= np.zeros((300,300),np.uint8)
# cv2.ellipse(ellipse,(150,150),(150,150),30,0,180,255,-1)
# cv2.imshow("Ellise",ellipse)
# cv2.waitKey(0)
# # cv2.destroyAllWindows()

#####image smoothing 
# import cv2
# img =cv2.imread("1.jpg",0)

# height,width =img.shape

# #Extract slop edge
# Sobel_x=  cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# Sobel_y=  cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

# cv2.imshow('orginal Image',img)

# cv2.imshow('sobel_x image',Sobel_x)
# cv2.imshow('sobel_y image',Sobel_y)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


#####
import cv2
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    cv2.imshow('our live sketch',frame)
    if cv2.waitKey(1)==13:
        break
cv2.release(0)
cv2.destroyAllWindows()