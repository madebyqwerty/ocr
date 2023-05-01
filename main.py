
import cv2, qrcode

debug_mode = False
image_scale = 0.2

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
        filtered_img = cv2.medianBlur(thresh_img, 101) #Filter showing approximate shape of the paper
        edges_img = cv2.Canny(filtered_img, 100, 200) #Edge detection
        contours, hierarchy = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])

        if h < (img.shape[0]/3) or w < (img.shape[1]/3): #If too small, probably poorly defined edges
            return img
        
        if debug_mode:
            cv2.imshow('Thresh', Image.resize(thresh_img, image_scale))
            cv2.imshow('Filter', Image.resize(filtered_img, image_scale))
            cv2.imshow('Edges', Image.resize(edges_img, image_scale))

        return img[y:y+h, x:x+w] #Crop

class Qr():
    """
    Qr code stuff
    """

    def create(data):
        """
        Make Qr code with data, return image
        """
        return qrcode.make(data)

    def process(img):
        """
        If needed rotates img and get qr data, returns img, data
        """
        qr_data, x, y = None, None, None
        qr_decoder = cv2.QRCodeDetector()
        data, bbox, _ = qr_decoder.detectAndDecode(img)

        if bbox is None:
            rotated_img = Image.flip(img)
            data, bbox, _ = qr_decoder.detectAndDecode(img)
            if bbox is not None:
                qr_data = data
                img = rotated_img
                x, y = bbox[0][0] #qrcode cords

        else: 
            x, y = bbox[0][0] #qrcode cords
            qr_data = data

        if x > img.shape[1]/2 or y > img.shape[0]/2: #if not in top right corner, flip it
            img = Image.flip(img)

        if qr_data is not None:
            return img, qr_data
        
        raise NotImplementedError

class Engine():
    """
    Primary functions
    """

    def process(file):
        """
        Image processing for the required data
        """
        preprocessed_img = Engine.image_preprocessing(file)
        img, qr_data = Qr.process(preprocessed_img)

        print(qr_data)

        cv2.imshow('Cropped, filtered Image', Image.resize(img, image_scale))
        cv2.waitKey(0) #Q for closing the window
        cv2.destroyAllWindows()

    def image_preprocessing(file):
        """
        Basic image processing
        """
        input_img = cv2.imread(file)
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) #Converts to shades of gray
        processed_img = cv2.medianBlur(Image.crop(gray_img), 3)
        ret, processed_img = cv2.threshold(processed_img, 110, 235, cv2.THRESH_BINARY) #Convert to binary image

        if processed_img.shape[0] > processed_img.shape[1]: 
            processed_img = Image.rotate(processed_img) #Turn if needed

        #TODO: Image.flip() if needed

        if debug_mode:
            cv2.imshow('Input', Image.resize(input_img, image_scale))

        return processed_img

if "__main__" == __name__:
    debug_mode = True
    Engine.process(f"TestImg/img1.jpg")
    Engine.process(f"TestImg/img2.jpg")
    Engine.process(f"TestImg/img3.jpg")