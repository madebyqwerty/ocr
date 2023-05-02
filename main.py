
import cv2, qrcode, datetime

debug_mode = False
image_scale = 0.2

class QRCodeError(Exception):
    pass

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
        _, thresh_img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) #Convert to binary image
        filtered_img = cv2.medianBlur(thresh_img, 81) #Filter showing approximate shape of the paper
        edges_img = cv2.Canny(filtered_img, 100, 200) #Edge detection
        contours, _ = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        
        if debug_mode: 
            #cv2.imshow('Thresh', Image.resize(thresh_img, image_scale))
            cv2.imshow('Edges', Image.resize(edges_img, image_scale))

        if h < (img.shape[0]/3) or w < (img.shape[1]/3): #If too small, probably poorly defined edges
            return img

        return img[y:y+h, x:x+w] #Crop

class Qr():
    """
    Qr code stuff
    """

    def create(version:str, teacher:str, class_id:str):
        """
        Make Qr code with data, return image
        """
        data = {
            "version": version,
            "create_date": datetime.datetime.now().strftime("%d.%m.%Y"),
            "teacher": teacher,
            "class_id": class_id
        }
        return qrcode.make(data)

    def process(img):
        """
        If needed rotates img and get qr data, returns img, data
        """
        qr_data, x, y = None, None, None

        _, binary_img = cv2.threshold(img, 120, 220, cv2.THRESH_BINARY) #Convert to binary image

        qr_decoder = cv2.QRCodeDetector()
        data, bbox, _ = qr_decoder.detectAndDecode(binary_img)

        if bbox is None: #If qr not decoded try flip
            rotated_img = Image.flip(binary_img)
            data, bbox, _ = qr_decoder.detectAndDecode(rotated_img)
            if bbox is not None:
                qr_data = data
                img = Image.flip(img)
                x, y = bbox[0][0] #qrcode cords

        else: 
            x, y = bbox[0][0] #qrcode cords
            qr_data = data

        if qr_data is not None:
            if x > img.shape[1]/2 or y > img.shape[0]/2: #if not in top right corner, flip it
                img = Image.flip(img)

            #return img, qr_data
        
        raise QRCodeError("QRCode is not readable") #No readable qrcode on img

class Engine():
    """
    Primary functions
    """

    def process(file):
        """
        Image processing for the required data
        """
        preprocessed_img = Engine.image_preprocessing(file)
        img, qr_data = Qr.process(preprocessed_img) #Rotate img if needed
        
        print(qr_data)
        
        _, img = cv2.threshold(img, 125, 210, cv2.THRESH_BINARY) #binary img (shades bug)
        # Shades bug --> binary is readable, depending on light
        # I can try change values if OCR cannot read correctly

        cv2.imshow('Processed', Image.resize(img, 0.3)) #image_scale
        cv2.waitKey(0) #Q for closing the window
        cv2.destroyAllWindows()

    def image_preprocessing(file):
        """
        Basic image processing
        """
        input_img = cv2.imread(file)
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) #Converts to shades of gray
        processed_img = cv2.medianBlur(Image.crop(gray_img), 3)

        if processed_img.shape[0] > processed_img.shape[1]: 
            processed_img = Image.rotate(processed_img) #Turn if needed

        if debug_mode: cv2.imshow('Input', Image.resize(input_img, image_scale))

        return processed_img

if "__main__" == __name__:
    debug_mode = True
    Engine.process(f"TestImg/img1.jpg")
    Engine.process(f"TestImg/img2.jpg")
    Engine.process(f"TestImg/img3.jpg")