
# Importing all the libararies

import cv2
import numpy as np
from PIL import Image, ImageChops
from scipy.ndimage import interpolation as inter
import os
import imutils
from skimage.filters import threshold_local
import operator
import functions_for_api


# required function from pyimagesearch for getting perspective view
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


# required function from pyimagesearch for getting perspective view
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


# required function from pyimagesearch for sorting contours
def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


# Function to detect the measurement page from image and preprocess for dewarping and making as a scanned document
def PreProcess(image_path, temp_file_dir_path):
    image = cv2.imread(image_path)  # Read the image
    ratio = image.shape[0] // 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converting into greyscale

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    edged = cv2.Canny(gray, 75, 200)  # Detects the edges in the image

    # FOR DEBUGGING
    # show the original image and the edge detected image
    # cv2.imwrite("./Temp/Image.jpg", edged)
    # cv2.imshow("Edged", edged)
    # cv2.imwrite("Edged.jpg", edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # This method helps to find the different shapes presented in the image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    # Sort all the contours by area wise and take first 5 areas which have larger area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # loop over the contours
    for c in cnts:
            # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
# Draw contours in image to see that we have detected our screent or not.It will make a border of green color over our sheet.
    try:

        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        cv2.imwrite(temp_file_dir_path+'Outline.jpg', image)
    except:
        print("Page is not detected!")

    # The value of variable ratio is set as image height divided by 500 because the image is rescaled to height 500
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # FOR DEBUGGING
    # cv2.imwrite("./Temp/warped.jpg",warped)

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

 # Thresholding our image if the color intensity is greater then 127, it will assign as 255 else all the pixel will be zero.
    T = threshold_local(warped, 127, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255

# Write the image into specific location
    cv2.imwrite(temp_file_dir_path+'img_for_box_extraction.jpg', warped)


# function to create review image
def review_image_creater(image_path, review_image_dir_path, image_name):
    image = cv2.imread(image_path)
    ratio = image.shape[0] // 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    edged = cv2.Canny(gray, 75, 200)

    # FOR DEBUGGING
    # show the original image and the edge detected image
    # cv2.imwrite("./Temp/Image.jpg", edged)
    # cv2.imshow("Edged", edged)
    # cv2.imwrite("Edged.jpg", edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(1)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # print(1)
    # loop over the contours
    for c in cnts:
            # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:

            screenCnt = approx
            break
    try:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    
        cv2.imwrite(review_image_dir_path+'Outline.jpg', image)
        x = 1
    except:
        x = 0

    return x
# selects the boxes that might contain the required information and adds them to cropped folder


def box_extraction(img_for_box_extraction_path, cropped_dir_path):

    img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image
    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin  # Invert the image

    hor_len = np.array(img).shape[1]//80  # Defining a kernel length
    # A verticle kernel of (1 X hor_len), which will detect all the verticle lines from the image.
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, hor_len))
    # A horizontal kernel of (hor_len X 1), which will help to detect all the horizontal line from the image.
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_len, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, kernel1, iterations=2)
    img_tmp1 = cv2.dilate(img_temp1, kernel1, iterations=3)
    # FOR DEBUGGING
    # cv2.imwrite("./Temp/verticle.jpg",img_tmp1)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, kernel2, iterations=2)
    img_tmp2 = cv2.dilate(img_temp2, kernel2, iterations=3)
    # FOR DEBUGGING
    # cv2.imwrite("./Temp/hori.jpg",img_tmp2)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(img_tmp1, alpha, img_tmp2, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    # cv2.imwrite("img_final_bin.jpg",img_final_bin)
    # Find contours for image, which will detect all the boxes
    im2, contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    idx = 0
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)

        # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        if (w > 80 and h > 20) and w > 3*h:
            idx += 1
            new_img = img[y:y+h, x:x+w]
            cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)

    # For Debugging
    # Enable this line to see all contours.
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    # cv2.imwrite("./Temp/img_contour.jpg", img)


# run on folder cropped and further processing the images and saving them to folder img_white
def to_white(cropped_dir_path, img_white_dir_path):
    # Path3 = './binary/'
    # files3 = os.listdir(Path3)
    # Fetch all the items from "/cropped" folder
    cropped_images_paths = os.listdir(cropped_dir_path)

    for single_cropped_image_path in cropped_images_paths:
        # Read all the images
        img = cv2.imread(cropped_dir_path+single_cropped_image_path, 0)
        img = np.array(img)

        if ((img.shape[1]//img.shape[0]) > 2 and img.shape[1] > 100 and img.shape[0] > 20):
            img = cv2.resize(img, (img.shape[1], img.shape[0]))
            # Thresholding image
            ret, gray = cv2.threshold(img, 165, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cnt = 0
            # If image is totally white and there is nothing information present in it then leave those image and
            # save only images with infomation in "img_white" folder.
            for i in range(0, gray.shape[0]):
                for j in range(0, gray.shape[1]):
                    if (gray[i][j] == 255):
                        cnt = cnt+1
            if(cnt >= (gray.shape[1]*gray.shape[0]//(1.02))):
                pass
            else:
                # gray = Image.fromarray(gray)
                # new_gray = rotate_image(gray)
                new_gray = np.array(gray)
                cv2.imwrite(img_white_dir_path+single_cropped_image_path, gray)
        else:
            pass

    # for name in files3:
    #     img = cv2.imread(Path3+name,0)
    #     img1 = 255 - img
    #     cv2.imwrite("./img_white/"+name,img1)


# RUNS ON FOLDER img_white and segments the digits and adding the segmented parts to folder mser
def Segmentation(image_path, mser_dir_path):
    # This value helps split the image in 2 at the boundary and the value has been determined by testing on various cells
    # and can be updated. If cells and serial numbers are not splitting properly 2 possible alternates could be setting it to 105(tested for 13 to 16mp camera)
    # and extracted_cell.shape[0]*1.35
    # if there is error in serial number  or the first digit of a number a possible error could be that this value is a bit off

    extracted_cell = cv2.imread(image_path, 0)

    extracted_cell = np.array(extracted_cell)
    (thresh, extracted_cell) = cv2.threshold(
        extracted_cell, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Commented function is to remove border in an image.
    # extracted_cell = Image.fromarray(extracted_cell)
    # extracted_cell = remove_border(extracted_cell)
    # Resize the image into (480,20)
    extracted_cell = cv2.resize(extracted_cell, (480, 120))
    # Cut 3-3 pixel from every side so it will helpful to detect digit if the digit is touching the border,
    # so we can divide the digit and border.
    extracted_cell = extracted_cell[3:117, 3:477]

    extracted_cell = np.pad(extracted_cell, ((3, 3), (3, 3)), 'maximum')
    # extracted_cell is updated to cell without serial number
    # FOR DEBUGGING
    #cv2.imwrite('./Temp/Debugging/extracted_cells/'+str(idx)+'_cell_info'+ '.png',new_img)
    cut_value = int(extracted_cell.shape[1]*0.2)
    extracted_cell_serial_no = extracted_cell[:, 0:cut_value-10]

    extracted_cell = extracted_cell[:, cut_value+10:480]
    # print extracted_cell.shape
    extracted_cell = cv2.resize(extracted_cell, (460, 100))
    extracted_cell = np.pad(extracted_cell, ((10, 10), (10, 10)), 'maximum')
    # extracted_cell is updated to cell's serial number
    # FOR DEBUGGING
    #cv2.imwrite('./Temp/Debugging/extracted_cells/'+str(idx)+'_cell_serial_no'+ '.png',new_img)
    # print extracted_cell.shape
    (thresh, extracted_cell) = cv2.threshold(
        extracted_cell, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    half = extracted_cell.shape[1]//2
    kernel = np.ones((3, 1), np.uint8)
    extracted_cell_serial_no_temp = cv2.erode(extracted_cell_serial_no, kernel, iterations=1)

    # detect regions in gray scale image
    im2, contours, hierarchy = cv2.findContours(
        extracted_cell, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    (contours, boundingBoxes) = sort_contours(contours, method="left-to-right")


    #detect regions in image which has serial_no
    im2, contours_serial_no, hierarchy = cv2.findContours(
        extracted_cell_serial_no_temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    (contours_serial_no, boundingBoxes) = sort_contours(contours_serial_no, method="left-to-right")

    temp = 0
    mask = np.zeros((extracted_cell.shape[0], extracted_cell.shape[1], 1), dtype=np.uint8)
    idx = 0
    idx1 = 0
    idx2 = 0
    for c in contours:
        #Drawing a bounding rectangle over contour points
        x, y, w, h = cv2.boundingRect(c)

        #Extract contours 
        new_img = extracted_cell[y:y+h, x:x+w]
        
        # If there is a contour which contains line then detect it as a line_(index) 
        if((y > 30 and y < 84) and (abs(w-h) > 70 or w//h >= 6)):
         
            new_img = extracted_cell[y:y+h, x:x+w]
            idx2 = idx2+1
            cv2.imwrite(mser_dir_path+'line_'+str(idx2) + '.png', new_img)

        #Ignoring contours which has noise
        if(abs(w-h) > 70 or w//h >= 6):
            pass
        #Noise ignorance
        elif(w >= half):
            pass
        #Segmentation of numbers and dots
        else:
            new_img = extracted_cell[y:y+h, x:x+w]

            # cv2.imwrite('new_crop/1_'+str(idx)+ '.png',new_img)
            # print temp

            if(h > 5 and w > 5):
                # x+w > temp, so it will not detect the inner contours of any contour
                if(x+w >= temp):
                    #Dot detection 
                    if((abs(w-h) <= 11 and h < 22) or w/float(h) >= 1.9):
                        idx = idx+1
                        new_img = 255 - new_img
                        cv2.imwrite(mser_dir_path+'dot_'+str(idx) + '.png', new_img)

                        temp = x+w

                    else:
                        #Number detection and segmentation
                        idx = idx+1
                        new_img = 255 - new_img
                        cv2.imwrite(mser_dir_path+'number_'+str(idx) + '.png', new_img)

                        temp = x+w

            else:
                pass
            # cv2.imwrite('mser/number_'+str(idx)+ '.png',new_img)
            # cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
    
    #Serial number detection
    idx = 0
    half = extracted_cell_serial_no.shape[1]//2
    temp = 0
    for c in contours_serial_no:
        x, y, w, h = cv2.boundingRect(c)
        #noise reduction
        if(w >= half):
            pass
        elif(abs(w-h) > 55 or w//h >= 6):
            pass
        #Serial number segmentation into 'mser' folder
        elif(h > 15 and w > 15):
            if(x+w >= temp):
                new_img = extracted_cell_serial_no[y:y+h, x:x+w]

                idx = idx+1
                new_img = 255 - new_img
                cv2.imwrite(mser_dir_path+'serial_'+str(idx) + '.png', new_img)
                temp = x+w
        else:
            pass
            # cv2.imwrite('mser/number_'+str(idx)+ '.png',new_img)
            # cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

#Trains the dataset using Singular vector decomposition
def Train_data(dataset_emp_dir_path, dataset_emp_dir_files):
    images = []
    mean_images = []
    for name in dataset_emp_dir_files:
        #Read all the images from dataset path
        dataset_image = cv2.imread(dataset_emp_dir_path+name)
        dataset_image = cv2.cvtColor(dataset_image, cv2.COLOR_BGR2GRAY)
        #Resize all the images into (45,55)
        dataset_image = cv2.resize(dataset_image, (45, 55), interpolation=cv2.INTER_AREA)
        # cv2.imwrite('./training/'+name, temp)
        #Append all the images into  one matrix [images] in vectorized format.
        images.append(dataset_image.flatten())
        # cv2.imwrite("./test_data/"+name, temp)
    images = np.array(images)
    #Find mean of the images and substract it from the original matrix to normalize the dataset.
    mean_images = np.mean(images)
    images = images-mean_images
    images = images.T
    
    #Input images into SVD function this will give U,S,V matrices,
    u, s, v = np.linalg.svd(images, full_matrices=False)

    #Compute the eigen vectors by multiplying image matrix with U.
    eigen_vectors_all_images = np.empty(shape=(u.shape[0]*u.shape[1], u.shape[1]),  dtype=np.int8)
    vector_temp = np.empty(shape=(u.shape[0], u.shape[1]),  dtype=np.int8)

    for i in range(images.shape[1]):

        for c in range(u.shape[1]):
            #Multiplication of all the images with U vector
            vector_temp[:, c] = images[:, i] * u[:, c]

        eigen_vectors = np.array(vector_temp, dtype='int8').flatten()
        eigen_vectors_all_images[:, i] = eigen_vectors[:]

    # np.save('u.npy',u)
    # np.save('s.npy',s)
    # np.save('v.npy',v)
    # np.save('images.npy',images)
    # np.save('mean_images.npy',mean_images)
    # np.save('feature_vector.npy',eigen_vectors_all_images)
    #Return the U,S,V vectors and mean and eigen  vectors.
    return u, s, v, images, mean_images, eigen_vectors_all_images


#Reads all the images from 'mser' folder and predicts numbers using training set.
def Test_data(u, s, v, images, mser_dir_path, mser_dir_files, mean_images, eigen_vectors_all_images, dataset_emp_dir_path):
    
    #Declaration of variables
    num_flag = [] #array for storing predicted numbers
    serial_flag = [] #Array for storing predicted serial_numbers.

    flag_width = 0 #Flags for detecting dots in length.
    flag_length = 0 #Flag for detecting dot in width.

    length_final = []
    width_final = []
    line = [0, 0]
    out_files = []
    idx = 0
    file_temp = []
    mser_dir_sorted = []
    serial_no = 0


    #Sorting the mser file directory to get input in sorted way.
    mser_dir_files.sort(key=lambda f: os.path.splitext(f)[0])

    for name in mser_dir_files:
        if 'serial' in name:
            mser_dir_sorted.append(str(name))



    for name in mser_dir_files:
        if 'number' in name:
            file_temp.append(int((name.split('_')[1]).split('.')[0]))
    file_temp.sort()
    for name in file_temp:
        mser_dir_sorted.append("number_"+str(name)+'.png')

    for name in mser_dir_files:
        if 'dot' in name:
            mser_dir_sorted.append(str(name))
        # if 'serial' in name:
        #     mser_dir_sorted.append(str(name))
        if 'line' in name:
            mser_dir_sorted.append(str(name))



    #Read all the input files from "mser" in sorted order.
    for name in mser_dir_sorted:
        serial_no = 0

        #If there is any line in the box then it will detect the line.
        if os.path.isfile(os.path.join(mser_dir_path, name)) and 'line' in name:

            line_name = name.split('_')[1]
            line_index = int(line_name.split('.')[0])
            if(int(line_index) > 0):

                if (int(line_index) == 1):
                    line[int(line_index-1)] = 1
                elif(int(line_index) == 2):
                    line[int(line_index-1)] = 1

        else:
            pass
    #     print length,width
        #If there is any dot file, then it will append the location of dot into length_flag and width_flag.
        if os.path.isfile(os.path.join(mser_dir_path, name)) and 'dot' in name:

            dot_name = name.split('_')[1]
            dot_index = int(dot_name.split('.')[0])
            #Appending the location of dot into flags.
            if(int(dot_index) > 0):

                if (int(dot_index) == 2):
                    # length[int(new_i-1)] = '.'
                    flag_length = dot_index
                elif(int(dot_index) == 3):
                    # length[int(new_i-1)] = '.'
                    flag_length = dot_index
                elif(int(dot_index) == 4):
                    # length[int(new_i-1)] = '.'
                    flag_length = dot_index

                else:
                    pass

                if (int(dot_index) == 6 or int(dot_index) == 5):
                    # width[1] = '.'
                    flag_width = dot_index
                elif (int(dot_index) == 7):
                    # width[1] = '.'
                    flag_width = dot_index
                elif (int(dot_index) == 8):
                    # width[1] = '.'
                    flag_width = dot_index
                elif(int(dot_index) == 9):
                    # width[2] = '.'
                    flag_width = dot_index
                elif(int(dot_index) == 10):
                    # width[3] = '.'
                    flag_width = dot_index
                else:
                    pass
        else:
            pass
    #     print length,width

        #Recognition for numbers using SVD.
        if os.path.isfile(os.path.join(mser_dir_path, name)) and 'number' in name:
            improving_idx = 0
            improving_dir_files = os.listdir("./Train_set/")
            improving_dir_files.sort(key=lambda f: os.path.splitext(f)[0])




            #Read test image
            test_img = np.array(cv2.imread(mser_dir_path+name))
            test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            #Resize it into (45,55)
            test = cv2.resize(test, (45, 55), interpolation=cv2.INTER_AREA)

            # cv2.imwrite('./all/'+name+'_'+str(idx)+'.png',test)
            idx = idx+1

            img = test.reshape(1, -1)

            #Substract the mean to normalize it.
            img = img-mean_images

            img = img.T
        #     print img[:][50]

            #Find the eigen vector of the test image by multiplying it with U matrix.

            test_x = np.empty(shape=(u.shape[0], u.shape[1]),  dtype=np.int8)
        #     print test_x.shape

            for col in range(u.shape[1]):
                test_x[:, col] = img[:, 0] * u[:, col]

            #Vectorize the eigen vector.
            eigen_vector_test_image = np.array(test_x, dtype='int8').flatten()

            #Find the difference between eigen vector of all images and eigen vector of test image to 
            #get the difference.
            object_difference = np.empty(shape=(u.shape[0]*u.shape[1], u.shape[1]))

            for col in range(u.shape[1]):
                object_difference[:, col] = eigen_vectors_all_images[:, col] - eigen_vector_test_image[:]

            #Store the norms of each vector in object_vector in ans, Sort the ans matrix, the index of the minimum
            #value of the ans will be our predicted digit.    
            ans = np.empty(shape=(u.shape[1],))

            for c in range(object_difference.shape[1]):
                ans[c] = np.linalg.norm(object_difference[:, c])

      
            ans_temp = np.copy(ans)
            #sorting the ans.
            ans_temp.sort()
            check = ans_temp[0]
        #     print check

            index = 0
            #Fetch index of the minimum value
            for i in range(ans.shape[0]):
                if check == ans[i]:

                    index = i

                    break
            #Find the indexed image from the dataset to map the number with training dataset number.        
            folder_tr = './dataset_new/'
            i = 0

            for filename in os.listdir(dataset_emp_dir_path):
                predicted_number_from_dataset = filename.split("_")[0]

                if index == i:

                    number_name = name.split('_')[1]
                    number_index = int(number_name.split('.')[0])
                    out_files.append(number_index)
                    #Append all the predicted numbers in "training set"  as predicted_numer_index_serial_no.
                    for dir_files in improving_dir_files:
                        if(dir_files.split("_")[0] == predicted_number_from_dataset):

                            if(int((dir_files.split("_")[1]).split(".")[0]) > improving_idx):
                                improving_idx = int((dir_files.split("_")[1]).split(".")[0])

                    improving_idx = improving_idx+1
                    serial_no = predict_number(serial_flag)
                    try:
                        if(int(serial_no)>0 and int(serial_no)< 90):
                            cv2.imwrite("./Train_set/"+str(predicted_number_from_dataset)+"_"+str(improving_idx)+"_"+str(serial_no)+".png",test_img)
                    except:
                        pass
                       #insert the number into num_flag.
                    b = num_flag.insert(int(number_index), predicted_number_from_dataset)
                    break

                else:
                    i = i+1

        #The same algorithm will be applied to predict the serial numbers.
        if os.path.isfile(os.path.join(mser_dir_path, name)) and 'serial' in name:

            test = np.array(cv2.imread(mser_dir_path+name))
            test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
            test = cv2.resize(test, (45, 55), interpolation=cv2.INTER_AREA)

            img = test.reshape(1, -1)
            img = img-mean_images

            img = img.T
     

            test_x = np.empty(shape=(u.shape[0], u.shape[1]),  dtype=np.int8)


            for col in range(u.shape[1]):
                test_x[:, col] = img[:, 0] * u[:, col]

            
            eigen_vector_test_image = np.array(test_x, dtype='int8').flatten()
            #Find eigen vector of the  test image
            object_difference = np.empty(shape=(u.shape[0]*u.shape[1], u.shape[1]))

            for col in range(u.shape[1]):
                object_difference[:, col] = eigen_vectors_all_images[:, col] - eigen_vector_test_image[:]
            ans = np.empty(shape=(u.shape[1],))

            for c in range(object_difference.shape[1]):
                ans[c] = np.linalg.norm(object_difference[:, c])
          
            ans_temp = np.copy(ans)

            ans_temp.sort()
            check = ans_temp[0]
        #     print check

            index = 0

            for i in range(ans.shape[0]):
                if check == ans[i]:

                    index = i

                    break
            folder_tr = './dataset_new/'
            i = 0

            for filename in os.listdir(dataset_emp_dir_path):
                predicted_srnumber_from_dataset = filename.split("_")[0]

                if index == i:
                    serial_name = name.split('_')[1]
                    serial_index = int(serial_name.split('.')[0])
                    # out_files1.append(new_i)
                    b = serial_flag.insert(int(serial_index), predicted_srnumber_from_dataset)
                    break

                else:
                    i = i+1


    #Append all the numbers in final_num_flag.
    final_num_flag = num_flag
  

    #Insert decimal points for length
    if(flag_length!= 0):
        final_num_flag.insert(int(flag_length-1), ".")

    #Insert decimal points for width
    if(flag_width!= 0):
        final_num_flag.insert(int(flag_width-1), ".")
 

    #Return the predicted numbers,serial_number and line flag.  
    return final_num_flag,serial_flag, line


#Function for predicting number from the array.
def predict_number(arr):
    result = ''
    try:
        for element in arr:
            result += str(element)

    except:
        result = 0
    return result

#Function for clearing all the files in the given directory path.
def clear_dir(Path):
    for the_file in os.listdir(Path):
        file_path = os.path.join(Path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            pass


#Function for creating directories using given paths
def create_imp_dirs(temp_dir_path, cropped_dir_path, mser_dir_path, Extracted_box_dir_path, dataset_emp_dir_path, img_white_dir_path, debugging_dir_path):
    if not os.path.exists(temp_dir_path):
        os.makedirs(temp_dir_path)
    if not os.path.exists(cropped_dir_path):
        os.makedirs(cropped_dir_path)
    if not os.path.exists(mser_dir_path):
        os.makedirs(mser_dir_path)
    if not os.path.exists(Extracted_box_dir_path):
        os.makedirs(Extracted_box_dir_path)
    if not os.path.exists(img_white_dir_path):
        os.makedirs(img_white_dir_path)
    if not os.path.exists(debugging_dir_path):
        os.makedirs(debugging_dir_path)
    if not os.path.exists(dataset_emp_dir_path):
        os.makedirs(dataset_emp_dir_path)


#Function to clear all the directories
def clear_directories_before_start(temp_file_dir_path, cropped_dir_path, mser_dir_path, Extracted_box_dir_path, img_white_dir_path, debugging_dir_path):
    clear_dir(cropped_dir_path)
    clear_dir(mser_dir_path)
    clear_dir(Extracted_box_dir_path)
    clear_dir(img_white_dir_path)
    if os.path.exists(temp_file_dir_path+'Outline.jpg'):
        os.remove(temp_file_dir_path+'Outline.jpg')
    if os.path.exists(temp_file_dir_path+'img_for_box_extraction.jpg'):
        os.remove(temp_file_dir_path+'img_for_box_extraction.jpg')


#Function to sort all the files from the directories.
def sort_files_in_dir(files_in_dir):
    file_temp = []

    for file in files_in_dir:
        file_temp.append(int(file.split('.')[0]))
    file_temp.sort()

    files_in_dir_sorted = []
    for file in file_temp:
        files_in_dir_sorted.append(str(file)+'.png')

    return files_in_dir_sorted


#Main function to preprocess the input image and train the dataset
def OCR_start(image_path, cropped_dir_path, dataset_emp_dir_path, dataset_emp_dir_files, temp_dir_path, img_white_dir_path, temp_file_dir_path):

    #Call preprocess function to detect the measurement sheet and preprocess the sheet.
    PreProcess(image_path, temp_file_dir_path)
    #Call box-extraction() to extract all the boxes from the image.
    box_extraction(temp_file_dir_path+"img_for_box_extraction.jpg", cropped_dir_path)
    #Function to preprocess all the boxes.
    to_white(cropped_dir_path, img_white_dir_path)
    # u = np.load('u.npy')
    # s = np.load('s.npy')
    # v = np.load('v.npy')
    # images1 = np.load('images.npy')
    # mean_images1 = np.load('mean_images.npy')
    # fu1 = np.load('feature_vector.npy')
   
    #Function to train the dataset.
    u, s, v, images, mean_images, eigen_vectors_all_images = Train_data(dataset_emp_dir_path, dataset_emp_dir_files)
    return u, s, v, images, mean_images, eigen_vectors_all_images


#Function to segment the boxes and predicting numbers from the boxes
def OCR_main(image_path, img_white_dir_single_file, mser_dir_path, img_white_dir_path, u, s, v, images, mean_images,eigen_vectors_all_images, dataset_emp_dir_path):
    length_final = 0
    width_final = 0
    for the_file in os.listdir(mser_dir_path):
        file_path = os.path.join(mser_dir_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            pass
    #Calling segmentation() to segment numbers from all the boxes
    Segmentation(img_white_dir_path+img_white_dir_single_file, mser_dir_path)

    mser_dir_files = os.listdir(mser_dir_path)
    mser_dir_files.sort()
    
    #Call test_data() to perform recognition of all the segmented numbers and serial_numbers
    final_num_flag, serial_flag, line = Test_data(
        u, s, v, images, mser_dir_path, mser_dir_files, mean_images, eigen_vectors_all_images, dataset_emp_dir_path)

    # print(final_num_flag)
    # print(line)
    # print(serial_flag)
    # print()

    output = []
    #Flag to detect if there is line in the boxes. if there is any line in the box then, it will be sent
    # to copy upper box's data and if there is no line then it will further predict the number.
    flag = line[0]+line[1]

    #If there is no line in the box.
    if(flag == 0):

        for x in range(0, len(final_num_flag)):

            try:
                #Divide the number flag into two parts left of the multiplication sign 
               # is length and right is width
                if (final_num_flag[x] == 'X'):
                    length_final = final_num_flag[:x]
                    width_final = final_num_flag[x+1:]

                    break
            except Exception as e:
                print(e)
        #Predict numbers from the arrays
        len_ans = predict_number(length_final)
        wid_ans = predict_number(width_final)
        try:
            len_ans = float(len_ans)
            wid_ans = float(wid_ans)
        except:
            len_ans = 0
            wid_ans = 0

        # len_ans = predict_num(length)
        len_ans = round(len_ans, 2)
        # wid_ans = predict_num(width)
        wid_ans = round(wid_ans, 2)

        #Predict the serial number
        serial_no = predict_number(serial_flag)
        try:
            serial_no = int(serial_no)
        except:
            serial_no = 0

        #Calculating the area
        area = len_ans*wid_ans
        area = round(area, 2)
        #Returning output as arrray of 4 elements, [serial number,length,width,area]
        output = [serial_no, len_ans, wid_ans, area]
        print(output)
        return output
    #if there is any line in the box
    else:
        #predicting the serial number
        serial_no = predict_number(serial_flag)
        try:
            serial_no = int(serial_no)
        except:
            serial_no = 0
        print([serial_no, "copy above entry"])
        #Returning output as 2 elements, [serial_no, "copy above entry"]
        return [serial_no, "copy above entry"]


def Run_OCR(company_name, username, image_name):

    temp = image_name.split('.')
    del temp[-1]
    temp_name = '.'.join(temp)
    temp1 = image_name.split('.')
    filename = temp_name+'_'+username+'.'+temp1[-1]

    company_dir_path = './all_data/'+company_name+'/'
    temp_dir_path = './all_data/'+str(company_name)+'/Temp/'
    temp_file_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/'
    cropped_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/cropped/'
    mser_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/mser/'
    Extracted_box_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/Extracted_box/'
    dataset_dir_path = './all_data/' + str(company_name)+'/dataset/'
    dataset_emp_dir_path = './all_data/' + str(company_name)+'/dataset/'+'dataset_emp_'+username+'/'
    img_white_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/img_white/'
    debugging_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/Debugging/'
    image_dir_path = './all_data/'+str(company_name)+'/Images/'
    review_images_dir_path = './all_data/'+str(company_name)+'review_images'

    functions_for_api.create_temp_directory(company_name, filename, username)
    clear_directories_before_start(temp_file_dir_path, cropped_dir_path, mser_dir_path,
                                   Extracted_box_dir_path, img_white_dir_path, debugging_dir_path)

    #Reading the image from image path
    image_path = image_dir_path+filename
    dataset_emp_dir_files = os.listdir(dataset_emp_dir_path)

    #Call OCR_start() to preprocessing
    u, s, v, images, mean_images, eigen_vectors_all_images = OCR_start(
        image_path, cropped_dir_path, dataset_emp_dir_path, dataset_emp_dir_files, temp_dir_path, img_white_dir_path, temp_file_dir_path)

    img_white_dir_files = os.listdir(img_white_dir_path)
    # for sorting the files in directory
    img_white_dir_files_sorted = sort_files_in_dir(img_white_dir_files)

    final_output = list()
    #Call OCR_main() for getting the output.
    for img_white_dir_single_file in img_white_dir_files_sorted:
        output = OCR_main(image_path, img_white_dir_single_file, mser_dir_path,
                          img_white_dir_path, u, s, v, images, mean_images, eigen_vectors_all_images, dataset_emp_dir_path)

        #Append all the output of boxes in final_output
        if((output[0] <= 90 and output[0] > 0) and len(output) == 2):
            final_output.append(output)
            cv2.imwrite(Extracted_box_dir_path+str(output[0])+".png",
                        cv2.imread(img_white_dir_path+img_white_dir_single_file))

        elif((output[0] <= 90 and output[0] > 0) and output[3] != 0.0):
            final_output.append(output)
            cv2.imwrite(Extracted_box_dir_path+str(output[0])+".png",
                        cv2.imread(img_white_dir_path+img_white_dir_single_file))

    final_output = sorted(final_output, key=operator.itemgetter(0), reverse=False)

    #If there is array of [serial_no,"copy above entry"], then append the data of previous box 
    #this array.
    for i in range(0, len(final_output)):
        if(len(final_output[i]) == 2):
            final_output[i] = [int(final_output[i][0]), final_output[i-1][1],
                               final_output[i-1][2], final_output[i-1][3], 1]
        else:
            final_output[i] = [int(final_output[i][0]), final_output[i][1],
                               final_output[i][2], final_output[i][3], 0]
    # functions_for_api.delete_entire_directory(company_name, image_name, username)



    return final_output


def create_review_image(company_name, image_name):
    image_dir_path = './all_data/'+str(company_name)+'/Images/'
    review_image_dir_path = './all_data/'+str(company_name)+'/review_images/'
    image_path = image_dir_path+str(image_name)
    if os.path.exists(review_image_dir_path+image_name):
        os.remove(review_image_dir_path+image_name)
    review_image_checker = review_image_creater(image_path, review_image_dir_path, image_name)
    return review_image_dir_path+image_name, review_image_checker

    # os.remove(review_image_dir_path+image_name)
#
a=Run_OCR('WiseL','123','t_17.jpg')
print(a)
#
#


def training_data_initial(company_name, username, image_name):
    temp = image_name.split('.')
    del temp[-1]
    temp_name = '.'.join(temp)
    temp1 = image_name.split('.')
    filename = temp_name+'_'+username+'.'+temp1[-1]

    training_emp_image_dir_path = './all_data/'+company_name+'/'+'Training/'+username+'/'+filename+'/'
    company_dir_path = './all_data/'+company_name+'/'
    temp_dir_path = './all_data/'+str(company_name)+'/Temp/'
    temp_file_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/'
    cropped_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/cropped/'
    mser_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/mser/'
    Extracted_box_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/Extracted_box/'
    dataset_dir_path = './all_data/' + str(company_name)+'/dataset/'
    dataset_emp_dir_path = './all_data/' + str(company_name)+'/dataset/'+'dataset_emp_'+username+'/'
    img_white_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/img_white/'
    debugging_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/Debugging/'
    image_dir_path = './all_data/'+str(company_name)+'/Images/'
    review_images_dir_path = './all_data/'+str(company_name)+'review_images'

    functions_for_api.create_temp_directory(company_name, filename, username)
    clear_directories_before_start(temp_file_dir_path, cropped_dir_path, mser_dir_path,
                                   Extracted_box_dir_path, img_white_dir_path, debugging_dir_path)

    image_path = image_dir_path+filename
    dataset_emp_dir_files = os.listdir(dataset_emp_dir_path)

    PreProcess(image_path, temp_file_dir_path)
    box_extraction(temp_file_dir_path+"img_for_box_extraction.jpg", cropped_dir_path)

    to_white(cropped_dir_path, img_white_dir_path)

    img_white_dir_files = os.listdir(img_white_dir_path)
    # for sorting the files in directory
    img_white_dir_files_sorted = sort_files_in_dir(img_white_dir_files)
    file_number = 1

    for img_white_dir_single_file in img_white_dir_files_sorted:
        Segmentation(img_white_dir_path+img_white_dir_single_file, mser_dir_path)
        mser_dir_files = os.listdir(mser_dir_path)
        for file in mser_dir_files:

            if 'number' in file:

                temporary = cv2.imread(mser_dir_path+file)

                cv2.imwrite(training_emp_image_dir_path+str(file_number)+'.png', temporary)
                file_number = file_number+1


#
# company_name = 'WiseL'
# image_name = 't_7.jpg'
# username = 'kanan1297'
#
# temp = image_name.split('.')
# del temp[-1]
# temp_name = '.'.join(temp)
# temp1 = image_name.split('.')
# filename = temp_name+'_'+username+'.'+temp1[-1]
#
# company_dir_path = './all_data/'+company_name+'/'
# temp_dir_path = './all_data/'+str(company_name)+'/Temp/'
# temp_file_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/'
# cropped_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/cropped/'
# mser_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/mser/'
# Extracted_box_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/Extracted_box/'
# dataset_dir_path = './all_data/' + str(company_name)+'/dataset/'
# dataset_emp_dir_path = './all_data/' + str(company_name)+'/dataset/'+'dataset_emp_'+username+'/'
# img_white_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/img_white/'
# debugging_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/Debugging/'
# image_dir_path = './all_data/'+str(company_name)+'/Images/'
# review_images_dir_path = './all_data/'+str(company_name)+'review_images'
#
# functions_for_api.create_temp_directory(company_name, filename, username)
# clear_directories_before_start(temp_file_dir_path, cropped_dir_path, mser_dir_path,
#                                Extracted_box_dir_path, img_white_dir_path, debugging_dir_path)
#
# image_path = image_dir_path+filename
# dataset_emp_dir_files = os.listdir(dataset_emp_dir_path)
#
# u, s, v, images, mu, fu = OCR_start(
#     image_path, cropped_dir_path, dataset_emp_dir_path, dataset_emp_dir_files, temp_dir_path, img_white_dir_path, temp_file_dir_path)
#
# img_white_dir_files = os.listdir(img_white_dir_path)
# # for sorting the files in directory
# img_white_dir_files_sorted = sort_files_in_dir(img_white_dir_files)
#
# final_output = list()
#
# output = OCR_main(image_path, '9.png', mser_dir_path,
#                   img_white_dir_path, u, s, v, images, mu, fu, dataset_emp_dir_path)
#
# print(output)
