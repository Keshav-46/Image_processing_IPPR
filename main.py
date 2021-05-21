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
import cv2
img=cv2.imread("1.jpg")
cv2.imshow('orginal',img)
cv2.waitKey(0)
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.destroyAllWindows()
cv2.imshow("Gray scale Imag",gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()