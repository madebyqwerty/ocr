
import cv2, qrcode, datetime, pytesseract

debug_mode = False
DEBUG_IMG_SCALE = 0.2

class QRCodeError(Exception):
    pass

class Image():
    """
    General image manipulation
    """
    def resize(img, size:float):
        """
        Resize the image
        """
        width = int(img.shape[1] * size)
        height = int(img.shape[0] * size)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    def convert_to_binary(img, val1:int, val2:int):
        """
        Converts image to binary
        """
        try: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except: None
        _, thresh_img = cv2.threshold(img, val1, val2, cv2.THRESH_BINARY) #Convert to binary image
        return thresh_img

    def crop_paper(img):
        """
        Crop the paper in image
        """
        thresh_img = Image.convert_to_binary(img, 120, 255)
        filtered_img = cv2.medianBlur(thresh_img, 81) #Filter showing approximate shape of the paper
        edges_img = cv2.Canny(filtered_img, 100, 200) #Edge detection
        contours, _ = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])

        if h < (img.shape[0]/3) or w < (img.shape[1]/3): #If too small, probably poorly defined edges
            return img

        return img[y:y+h, x:x+w] #Crop

    def crop_table(img):
        """
        get table from image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 9)
        gray_thresh = cv2.bilateralFilter(gray_thresh, 9, 75, 75)
        binary = Image.convert_to_binary(gray_thresh, 130, 255)

        filtered_img = cv2.medianBlur(binary, 3)
        inverted_img = cv2.bitwise_not(filtered_img)

        contours, _ = cv2.findContours(inverted_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        best_rect = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > max_area:
                max_area = area
                best_rect = (x, y, w, h)

        x, y, w, h = best_rect

        if debug_mode: 
            cv2.imshow('Table', Image.resize(binary[y:y+h, x:x+w], DEBUG_IMG_SCALE)) #image_scale

        return img[y-25:y+h+25, x-25:x+w+25]

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

        binary_img = Image.convert_to_binary(img, 130, 255)
        qr_decoder = cv2.QRCodeDetector()
        data, bbox, _ = qr_decoder.detectAndDecode(binary_img)

        if bbox is None: #If qr not decoded try flip
            rotated_img = cv2.rotate(binary_img, cv2.ROTATE_180)
            data, bbox, _ = qr_decoder.detectAndDecode(rotated_img)
            if bbox is not None:
                qr_data = data
                img =cv2.rotate(img, cv2.ROTATE_180)
                x, y = bbox[0][0] #qrcode cords

        else: 
            x, y = bbox[0][0] #qrcode cords
            qr_data = data

        if qr_data:
            if x > img.shape[1]/2 or y > img.shape[0]/2: #if not in top right corner, flip it
                img = cv2.rotate(img, cv2.ROTATE_180)

            return img, qr_data #Convert to dict
        
        raise QRCodeError("QRCode is not readable") #No readable qrcode on img

class OCR():
    """
    OCR processing
    """
    def img_processing(input_img):
        """
        Get raw text from image
        """
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        gray_thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 9)
        gray_thresh_img = cv2.bilateralFilter(gray_thresh_img, 9, 75, 75)
        gray_thresh_img = cv2.medianBlur(gray_thresh_img, 3)
        binary_img = Image.convert_to_binary(gray_thresh_img, 130, 255)
        edges_img = cv2.Canny(binary_img, 100, 200) #Edge detection

        text = pytesseract.image_to_string(edges_img, "ces") #sudo dnf install tesseract-langpack-ces tesseract
        text = text.replace("\n", ", ")

        if debug_mode: 
            cv2.imshow('OCR', Image.resize(edges_img, DEBUG_IMG_SCALE)) #image_scale

        return text #Test return

    def text_processing(text:str, qr_data:str):
        """
        Process raw text
        """
        text_list = text.split(" ")
        text_edited = []

        for i in text_list:
            if len(i) >= 3:
                text_edited.append(i)

        return text

class Engine():
    """
    Primary functions
    """
    def process(file):
        """
        Image processing for the required data
        """
        input_img = cv2.imread(file)
        paper_img = Image.crop_paper(input_img)
        filtered_img = cv2.medianBlur(paper_img, 3)

        if filtered_img.shape[0] > filtered_img.shape[1]: #Turn horizontal
            filtered_img = cv2.rotate(filtered_img, cv2.ROTATE_90_CLOCKWISE) 

        if debug_mode: 
            cv2.imshow('Input', Image.resize(filtered_img, DEBUG_IMG_SCALE))

        img, qr_data = Qr.process(input_img) #Get qr data, flip if needed
        
        print(qr_data)

        table_img = Image.crop_table(img)
        raw_data = OCR.img_processing(table_img)
        data = OCR.text_processing(raw_data, qr_data)

        print(data)

        if debug_mode:
            cv2.waitKey(0) #Q for closing the window
            cv2.destroyAllWindows()
    
if "__main__" == __name__:
    debug_mode = True

    #img = Qr.create("2855604082", "5755190332")
    #img.save("Qr.jpg")

    Engine.process(f"imgs/img1.jpg")
