
import cv2

class Image():
    """
    Jakákoli univerzální manipulace s obrázkem
    """

    def rotate(img):
        """
        Otočí obrázek o 90 stupňů
        """
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    def flip(img):
        """
        Otočí obrázek vzhůru nohama
        """
        return cv2.rotate(img, cv2.ROTATE_180)
    
    def resize(img, size:float):
        """
        Změní velikost obrázku
        """
        width = int(img.shape[1] * size)
        height = int(img.shape[0] * size)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    def cut_off(img):
        """
        Vyřízne z obrázku papír
        """
        filtered_img = cv2.medianBlur(img, 81) #Filtr zobrazující jen přibližný tvar papíru
        edges_img = cv2.Canny(filtered_img, 100, 200) #Detekuje hrany
        contours, hierarchy = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        return img[y:y+h, x:x+w] #Ořízne papír

class Engine():
    """
    Primární funkce
    """

    def process(file):
        """
        Proces spracování obrázku na požadovaná data
        """
        preprocessed_img = Engine.image_preprocessing(file)
        cv2.imshow('Cropped, filtered Image', Image.resize(preprocessed_img, 0.3))
        cv2.waitKey(0) #Q pro zavření
        cv2.destroyAllWindows()

    def image_preprocessing(file):
        """
        Základní zpracování obrázku
        """
        input_img = cv2.imread(file)
        if input_img.shape[0] > input_img.shape[1]: 
            input_img = Image.rotate(input_img) #Pokud je opotřeba, otočí na šířku

        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) #Převede na černobílé
        ret, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY) #Převede na binární

        working_img = cv2.medianBlur(Image.cut_off(thresh_img), 3)
        return working_img

if "__main__" == __name__:
    Engine.process("testImg.jpg")