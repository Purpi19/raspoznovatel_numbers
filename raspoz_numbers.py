import cv2
import pytesseract
import matplotlib.pyplot as plt


def open_img(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(image)
    plt.show()

    return image

def image_extract(img, image_haar_cascade):
    image_rects = image_haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in image_rects:
        vagon_img = img[y+15:y+h-0, x+10:x+w-10]

    return vagon_img

def enlarge_img(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    plt.axis('off')
    resized_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return resized_image

def main():
    image_rgb = open_img(img_path='C:/raspoznovanie_numbers/vagon_numbers/17.jpg')
    image_haar_cascade = cv2.CascadeClassifier('C:/raspoznovanie_numbers/haar_cascades/haarcascade_russian_plate_number.xml')

    number_extract_img = image_extract(image_rgb, image_haar_cascade)
    number_extract_img = enlarge_img(number_extract_img, 150)
    plt.imshow(number_extract_img)
    # plt.show()

    number_extract_img_gray = cv2.cvtColor(number_extract_img, cv2.COLOR_RGB2GRAY)
    plt.axis('off')
    plt.imshow(number_extract_img_gray, cmap='gray')
    plt.show()
    tessdata_dir_config = r'--tessdata-dir"C:/Program Files (x86)/Tesseract-OCR/tessdata"'
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
    text = pytesseract.image_to_string(
        number_extract_img_gray,
        config='--oem 3 --psm 13 '
    )
    print(text)


if __name__ == '__main__':
    main()



