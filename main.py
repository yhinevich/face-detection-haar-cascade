import cv2
import sys

# чтение аргументов командной строки
imagePath = sys.argv[1]

# чтение изображения и преобразование в оттенки серого, так как
# распознавание объектов по яркости даёт лучшие результаты в сравнение
# с распознаванием по цвету
image = cv2.imread(imagePath)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# создание объекта faceCascade, который загрузит файл каскада Хаара с помощью метода cv2.CascadeClassifier
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# создание списка прямоугольников для всех обнаруженных лиц на изображении.
faces = faceCascade.detectMultiScale(
# указывает на использование объекта изображения OpenCV в оттенках серого
gray,
# этот параметр указывает скорость уменьшения размера изображения при каждом масштабе
scaleFactor=1.3,
# этот параметр указывает, сколько соседей (или обнаружений) должен иметь каждый прямоугольник-кандидат для его
# сохранения. Более высокое значение может привести к меньшему количеству ложных срабатываний, но слишком высокое
# значение может исключить истинные положительные результаты
minNeighbors=5,
# минимально возможный размер объекта в пикселях
minSize=(30, 30)
)
print("[INFO] Found {0} Faces!".format(len(faces)))
# отрисовка прямоугольников вокруг обнаруженных лиц
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
# запись результата распознавания в локальную память, булин для проверки успешности записи
status = cv2.imwrite(imagePath.replace(imagePath.split("\\")[-1],"")  + 'detected_' + imagePath.split("\\")[-1], image)
print("[INFO] Image written to filesystem: ", status)
