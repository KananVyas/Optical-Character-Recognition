from imutils import paths
import cv2


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def blur_detection(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = 50.0

    fm = variance_of_laplacian(image)
    # text = "Not Blurry"

    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < threshold:
        text = "Blurry"
    else:
        text = 'Not Blurry'
    return text, fm


# 	# show the image
# cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

# cv2.imshow("Image", image)

# key = cv2.waitKey(0)

def run_blur_detection(company_name, image_name):
    image_path = './all_data/'+company_name+'/Images/'+image_name
    text, fm = blur_detection(image_path)
    message = 'if fm value is above 200 it is probably not blurry otherwise it is blurry'
    return text, fm, message, image_path
