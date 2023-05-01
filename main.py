
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
        ret, thresh_img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) #Convert to binary image
        #cv2.imshow('Thresh', Image.resize(thresh_img, 0.15))
        filtered_img = cv2.medianBlur(thresh_img, 101) #Filter showing approximate shape of the paper
        #cv2.imshow('Filter', Image.resize(filtered_img, 0.15))
        edges_img = cv2.Canny(filtered_img, 100, 200) #Edge detection
        #cv2.imshow('Edges', Image.resize(edges_img, 0.15))
        contours, hierarchy = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])

        if w < 500 or h < 500: #If too small, probably poorly defined edges
            return img

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
        cv2.imshow('Cropped, filtered Image', Image.resize(preprocessed_img, 0.2))
        cv2.waitKey(0) #Q for closing the window
        cv2.destroyAllWindows()

    def image_preprocessing(file):
        """
        Basic image processing
        """
        input_img = cv2.imread(file)
        cv2.imshow('Input', Image.resize(input_img, 0.2))
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) #Converts to shades of gray
        processed_img = cv2.medianBlur(Image.crop(gray_img), 3)
        ret, processed_img = cv2.threshold(processed_img, 110, 235, cv2.THRESH_BINARY) #Convert to binary image

        if processed_img.shape[0] > processed_img.shape[1]: 
            processed_img = Image.rotate(processed_img) #Turn if needed

        #TODO: Image.flip() if needed

        return processed_img

if "__main__" == __name__:
    Engine.process(f"TestImg/img0.jpg")
    #Engine.process(f"TestImg/img1.jpg")
    #Engine.process(f"TestImg/img2.jpg")
    #Engine.process(f"TestImg/img3.jpg")
    #Engine.process(f"TestImg/img4.jpg")
    #Engine.process(f"TestImg/img5.jpg")