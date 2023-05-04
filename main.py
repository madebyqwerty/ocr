
import numpy as np
import cv2, qrcode, datetime, pytesseract

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
    
    def convert_to_gray(img):
        """
        Converts image to gray shades
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def convert_to_binary(img, val1, val2):
        """
        Converts image to binary
        """
        try: img = Image.convert_to_gray(img)
        except: None
        _, thresh_img = cv2.threshold(img, val1, val2, cv2.THRESH_BINARY) #Convert to binary image
        return thresh_img

    def crop(img):
        """
        Crops the image
        """
        thresh_img = Image.convert_to_binary(img, 120, 255)
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

    def create(teacher_id:str, class_id:str):
        """
        Make Qr code with data, return image
        """
        data = {
            "create_date": datetime.datetime.now().strftime("%d.%m.%Y"),
            "teacher_id": teacher_id,
            "class_id": class_id
        }
        return qrcode.make(data)

    def process(img):
        """
        If needed rotates img and get qr data, returns img, data
        """
        qr_data, x, y = None, None, None

        binary_img = Image.convert_to_binary(img, 120, 220)

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

            return img, qr_data
        
        raise QRCodeError("QRCode is not readable") #No readable qrcode on img

class OCR():
    """
    OCR processing
    """
    def process(input_img):
        """
        Get name and absence from image
        """

        # TODO: split verticaly to get name and days separately?
        # TODO: get name and absence

        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        gray_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 9)
        gray_thresh = cv2.bilateralFilter(gray_thresh, 9, 75, 75)
        img = Image.convert_to_binary(gray_thresh, 130, 255)

        text = pytesseract.image_to_string(img)
        print(text.replace("\n", ", "))

        if debug_mode: cv2.imshow('OCR', Image.resize(img, 0.25)) #image_scale

        return None, None #Name, absence

class Engine():
    """
    Primary functions
    """

    def process(file):
        """
        Image processing for the required data
        """
        input_img = cv2.imread(file)
        preprocessed_img = Engine.image_preprocessing(input_img)
        img, qr_data = Qr.process(preprocessed_img) #Rotate img if needed
        
        print(qr_data)

        OCR.process(img)

        #data = Engine.paper_processing(img)
        #print(data, "\n")

        if debug_mode:
            cv2.waitKey(0) #Q for closing the window
            cv2.destroyAllWindows()

    def image_preprocessing(input_img):
        """
        Basic image processing
        """
        processed_img = cv2.medianBlur(Image.crop(input_img), 3)

        if processed_img.shape[0] > processed_img.shape[1]: 
            processed_img = Image.rotate(processed_img) #Turn if needed

        if debug_mode: cv2.imshow('Input', Image.resize(input_img, image_scale))

        return processed_img
    
    def paper_processing(img):
        """
        Table processing
        """
        #binary_img = Image.convert_to_binary(input_img, 140, 255)
        #filtered_img = cv2.medianBlur(binary_img, 7)

        # TODO: Get lines on paper and for every part do OCR
        # TODO: for cut_img in img_list OCR.process(cut_img) --> to have option to set custom filters
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # adaptivní prahování s větším oknem
        gray_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 9)
        gray_thresh = cv2.bilateralFilter(gray_thresh, 9, 75, 75)

        binary = Image.convert_to_binary(gray_thresh, 130, 255)

        # detekce hran Cannyho algoritmem
        edges = cv2.Canny(binary, 50, 150, apertureSize=5)

        # detekce horizontálních linií pomocí Houghovy transformace
        lines = cv2.HoughLinesP(edges, rho=1, theta=1*np.pi/180, threshold=80, minLineLength=1800, maxLineGap=100)

        # vykreslení detekovaných linií do obrázku
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y1 + 25 > y2 and y1 - 25 < y2:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if debug_mode:
            cv2.imshow('Tresh', Image.resize(gray_thresh, 0.2)) #image_scale
            cv2.imshow('Edges', Image.resize(edges, 0.2)) #image_scale
            cv2.imshow('Processed', Image.resize(img, 0.2)) #image_scale

        return None

if "__main__" == __name__:
    debug_mode = True
    #img = Qr.create("2855604082", "5755190332")
    #img.save("Qr.jpg")
    Engine.process(f"TestImg/img2.jpg")
