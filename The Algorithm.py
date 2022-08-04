 #Importing the required external libraries 
import cv2
import numpy as np
import glob
from skimage import io

#loading the original images from the file 'Selected' where they are stored 
file_list = glob.glob(r'IRIS Dataset\SCAN\Selected\*.*')

my_list = []

#Reading IRIS Images and applying Laplacian of Guassian Filter on them 
for file in file_list:
    im = io.imread(file)
    my_list.append(im)
    output = im.copy()
    im2 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    Gaussian_filter = cv2.GaussianBlur(im2, (7,7),1)
    cv2.imshow('image', im2)
    im2 = cv2.Canny(im2, 80, 100, apertureSize=3)
    cv2.imshow("image2", im2)
    cv2.waitKey()
    cv2.destroyAllWindows()


#Reading IRIS Images again but individually this time 
img1 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\aevar2.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\bryanr4.bmp', cv2.IMREAD_GRAYSCALE) 
img3 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\chingycr3.bmp', cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\chongpkr1.bmp', cv2.IMREAD_GRAYSCALE) 
img5 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\christiner2.bmp', cv2.IMREAD_GRAYSCALE) 
img6 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\chualsr1.bmp', cv2.IMREAD_GRAYSCALE) 
img7 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\eugenehor1.bmp', cv2.IMREAD_GRAYSCALE) 
img8 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\fatmar1.bmp', cv2.IMREAD_GRAYSCALE) 
img9 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\fionar2.bmp', cv2.IMREAD_GRAYSCALE) 
img10 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\hockr2.bmp', cv2.IMREAD_GRAYSCALE) 
img11 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\kelvinr5.bmp', cv2.IMREAD_GRAYSCALE) 
img12 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\lecr3.bmp', cv2.IMREAD_GRAYSCALE) 
img13 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\liujwr4.bmp', cv2.IMREAD_GRAYSCALE) 
img14 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\loker1.bmp', cv2.IMREAD_GRAYSCALE) 
img15 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\lowyfr4.bmp', cv2.IMREAD_GRAYSCALE) 
img16 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\lpjr3.bmp', cv2.IMREAD_GRAYSCALE) 
img17 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\mahskr4.bmp', cv2.IMREAD_GRAYSCALE) 
img18 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\maranr1.bmp', cv2.IMREAD_GRAYSCALE) 
img19 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\masr2.bmp', cv2.IMREAD_GRAYSCALE) 
img20 = cv2.imread(r'C:\Users\aliha\Desktop\IRIS Dataset\SCAN\Selected\mazwanr5.bmp', cv2.IMREAD_GRAYSCALE)

#Designing the cicles of segmentation for image 1
circles = cv2.HoughCircles(img1, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output1 = img1.copy()
for (x, y, r) in circles:
    cv2.circle(output1, (x-33, y), r-42, (170, 100, 0), 2)
    cv2.circle(output1, (x-33, y), r-13, (0, 0, 255), 2)
    break
#Showing image 1 output after segmentation process 
cv2.imshow("Output 1", output1)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 2
circles = cv2.HoughCircles(img2, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output2 = img2.copy()
for (x, y, r) in circles:
    cv2.circle(output2, (x+51, y-37), r-70, (170, 100, 0), 2)
    cv2.circle(output2, (x+49, y-37), r-43, (0, 0, 255), 2)
    break
#Showing image 2 output after segmentation process 
cv2.imshow("Output 2", output2)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 3
circles = cv2.HoughCircles(img3, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output3 = img3.copy()
for (x, y, r) in circles:
    cv2.circle(output3, (x-35, y+20), r-70, (170, 100, 0), 2)
    cv2.circle(output3, (x-35, y+23), r-43, (0, 0, 255), 2)
    break
#Showing image 3 output after segmentation process 
cv2.imshow("Output 3", output3)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 4
circles = cv2.HoughCircles(img4, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output4 = img4.copy()
for (x, y, r) in circles:
    cv2.circle(output4, (x+35, y-9), r-60, (170, 100, 0), 2)
    cv2.circle(output4, (x+33, y-9), r-30, (0, 0, 255), 2)
    break
#Showing image 4 output after segmentation process 
cv2.imshow("Output 4", output4)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 5
circles = cv2.HoughCircles(img5, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output5 = img5.copy()
for (x, y, r) in circles:
    cv2.circle(output5, (x+10, y+5), r-70, (170, 100, 0), 2)
    cv2.circle(output5, (x+10, y+5), r-41, (0, 0, 255), 2)
    break
#Showing image 5 output after segmentation process 
cv2.imshow("Output 5", output5)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 6
circles = cv2.HoughCircles(img6, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output6 = img1.copy()
for (x, y, r) in circles:
    cv2.circle(output6, (x-22, y-40), r-42, (170, 100, 0), 2)
    cv2.circle(output6, (x-23, y-40), r-13, (0, 0, 255), 2)
    break
#Showing image 6 output after segmentation process 
cv2.imshow("Output 6", output6)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 7
circles = cv2.HoughCircles(img7, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output7 = img7.copy()
for (x, y, r) in circles:
    cv2.circle(output7, (x-6, y), r-80, (170, 100, 0), 2)
    cv2.circle(output7, (x-3, y-1), r-45, (0, 0, 255), 2)
    break
#Showing image 7 output after segmentation process 
cv2.imshow("Output 7", output7)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 8
circles = cv2.HoughCircles(img8, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output8 = img8.copy()
for (x, y, r) in circles:
    cv2.circle(output8, (x-2, y+4), r-71, (170, 100, 0), 2)
    cv2.circle(output8, (x-3, y+4), r-41, (0, 0, 255), 2)
    break
#Showing image 8 output after segmentation process 
cv2.imshow("Output 8", output8)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 9
circles = cv2.HoughCircles(img9, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output9 = img9.copy()
for (x, y, r) in circles:
    cv2.circle(output9, (x-76, y-100), r-60, (170, 100, 0), 2)
    cv2.circle(output9, (x-78, y-100), r-39, (0, 0, 255), 2)
    break
#Showing image 9 output after segmentation process 
cv2.imshow("Output 9", output9)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 10
circles = cv2.HoughCircles(img10, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output10 = img10.copy()
for (x, y, r) in circles:
    cv2.circle(output10, (x+83, y+62), r-36, (170, 100, 0), 2)
    cv2.circle(output10, (x+80, y+62), r-6, (0, 0, 255), 2)
    break
#Showing image 10 output after segmentation process 
cv2.imshow("Output 10", output10)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 11
circles = cv2.HoughCircles(img11, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output11 = img11.copy()
for (x, y, r) in circles:
    cv2.circle(output11, (x-49, y+3), r-82, (170, 100, 0), 2)
    cv2.circle(output11, (x-50, y+3), r-52, (0, 0, 255), 2)
    break
#Showing image 11 output after segmentation process 
cv2.imshow("Output 11", output11)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 12
circles = cv2.HoughCircles(img12, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output12 = img12.copy()
for (x, y, r) in circles:
    cv2.circle(output12, (x+29, y), r-75, (170, 100, 0), 2)
    cv2.circle(output12, (x+32, y), r-45, (0, 0, 255), 2)
    break
#Showing image 12 output after segmentation process 
cv2.imshow("Output 12", output12)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 13
circles = cv2.HoughCircles(img13, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output13 = img13.copy()
for (x, y, r) in circles:
    cv2.circle(output13, (x-60, y-22), r-76, (170, 100, 0), 2)
    cv2.circle(output13, (x-62, y-24), r-42, (0, 0, 255), 2)
    break
#Showing image 13 output after segmentation process 
cv2.imshow("Output 13", output13)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 14
circles = cv2.HoughCircles(img14, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output14 = img14.copy()
for (x, y, r) in circles:
    cv2.circle(output14, (x+10, y-37), r-47, (170, 100, 0), 2)
    cv2.circle(output14, (x+9, y-37), r-10, (0, 0, 255), 2)
    break
#Showing image 14 output after segmentation process 
cv2.imshow("Output 14", output14)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 15
circles = cv2.HoughCircles(img15, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output15 = img15.copy()
for (x, y, r) in circles:
    cv2.circle(output15, (x+30, y+10), r-27, (170, 100, 0), 2)
    cv2.circle(output15, (x+30, y+10), r-4, (0, 0, 255), 2)
    break
#Showing image 15 output after segmentation process 
cv2.imshow("Output 15", output15)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 16
circles = cv2.HoughCircles(img16, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output16 = img16.copy()
for (x, y, r) in circles:
    cv2.circle(output16, (x, y-3), r-53, (170, 100, 0), 2)
    cv2.circle(output16, (x-3, y-3), r-20, (0, 0, 255), 2)
    break
#Showing image 16 output after segmentation process 
cv2.imshow("Output 16", output16)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 17
circles = cv2.HoughCircles(img17, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output17 = img17.copy()
for (x, y, r) in circles:
    cv2.circle(output17, (x-17, y-30), r-57, (170, 100, 0), 2)
    cv2.circle(output17, (x-18, y-30), r-30, (0, 0, 255), 2)
    break
#Showing image 17 output after segmentation process 
cv2.imshow("Output 17", output17)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 18
circles = cv2.HoughCircles(img18, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output18 = img18.copy()
for (x, y, r) in circles:
    cv2.circle(output18, (x-28, y), r-55, (170, 100, 0), 2)
    cv2.circle(output18, (x-28, y), r-27, (0, 0, 255), 2)
    break
#Showing image 18 output after segmentation process 
cv2.imshow("Output 18", output18)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 19
circles = cv2.HoughCircles(img19, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output19 = img19.copy()
for (x, y, r) in circles:
    cv2.circle(output19, (x+81, y+42), r-74, (170, 100, 0), 2)
    cv2.circle(output19, (x+81, y+42), r-52, (0, 0, 255), 2)
    break
#Showing image 19 output after segmentation process 
cv2.imshow("Output 19", output19)
cv2.waitKey()
cv2.destroyAllWindows()

#Designing the cicles of segmentation for image 20
circles = cv2.HoughCircles(img20, cv2.HOUGH_GRADIENT, 10, 100,
                           param1 = 50, param2 = 50, minRadius=0, maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
output20 = img20.copy()
for (x, y, r) in circles:
    cv2.circle(output20, (x+21, y-20), r-48, (170, 100, 0), 2)
    cv2.circle(output20, (x+18, y-21), r-19, (0, 0, 255), 2)
    break
#Showing image 20 output after segmentation process 
cv2.imshow("Output 20", output20)
cv2.waitKey()
cv2.destroyAllWindows()

