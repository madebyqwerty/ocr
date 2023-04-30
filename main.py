
import cv2

class Image():
    """
    General image manipulation
    """

    def rotate(img):
        """
        Rotates the image 90 degrees
        """
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    def flip(img):
        """
        Turns the image upside down
        """
        return cv2.rotate(img, cv2.ROTATE_180)
    
    def resize(img, size:float):
        """
        Resize the image
        """
        width = int(img.shape[1] * size)
        height = int(img.shape[0] * size)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    def crop(img):
        """
        Crops the image
        """
        filtered_img = cv2.medianBlur(img, 81) #Filter showing approximate shape of the paper
        edges_img = cv2.Canny(filtered_img, 100, 200) #Edge detection
        contours, hierarchy = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        return img[y:y+h, x:x+w] #Crop

class Engine():
    """
    Primary functions
    """

    def process(file):
        """
        Image processing for the required data
        """
        preprocessed_img = Engine.image_preprocessing(file)
        cv2.imshow('Cropped, filtered Image', Image.resize(preprocessed_img, 0.3))
        cv2.waitKey(0) #Q for closing the window
        cv2.destroyAllWindows()

    def image_preprocessing(file):
        """
        Basic image processing
        """
        input_img = cv2.imread(file)
        if input_img.shape[0] > input_img.shape[1]: 
            input_img = Image.rotate(input_img) #Turn if needed

        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) #Converts to shades of gray
        ret, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY) #Convert to binary image

        working_img = cv2.medianBlur(Image.crop(thresh_img), 3)
        return working_img

if "__main__" == __name__:
    Engine.process("testImg.jpg")