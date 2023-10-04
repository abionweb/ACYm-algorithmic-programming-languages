import os
import cv2
import svgwrite as sw

class Images:
    def __init__(self):
        self.__next_key = 1
        self.__free_images = []
        self.__panorama_images = []

    def addImage(self, file_path, image):
        self.__free_images.append(Image(image, self.__next_key, file_path))
        self.__next_key += 1

    def process(self):
        if len(self.__free_images) == 0:
            print("empty directory")
            return
        self.__panorama_add_first_image()
        while len(self.__free_images) != 0:
            self.__panorama_add_next_image()
        self.__panorama_coordinate_normalization()
        self.__save_panorama_svg()

    def __getImage(self, key):
        for image in self.__free_images:
            if image.key == key:
                return image
        for image in self.__panorama_images:
            if image.key == key:
                return image

    def __panorama_add_first_image(self):
        image = self.__free_images[0]
        self.__panorama_images.append(image)
        self.__free_images.remove(image)
        image.panorama_x = 0
        image.panorama_y = 0

    def __panorama_add_next_image(self):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        distance = -1
        for panorama_image in self.__panorama_images:
            for image in self.__free_images:
                matches = matcher.match(panorama_image.des, image.des)
                matches = sorted(matches, key=lambda x: x.distance)
                if distance == -1 or distance > matches[0].distance:
                    distance = matches[0].distance
                    key = image.key
                    panorama_x = panorama_image.panorama_x + panorama_image.kp[matches[0].queryIdx].pt[0] - image.kp[matches[0].trainIdx].pt[0]
                    panorama_y = panorama_image.panorama_y + panorama_image.kp[matches[0].queryIdx].pt[1] - image.kp[matches[0].trainIdx].pt[1]
        image = self.__getImage(key)
        image.panorama_x = panorama_x
        image.panorama_y = panorama_y
        self.__free_images.remove(image)
        self.__panorama_images.append(image)
    def __panorama_coordinate_normalization(self):
        min_x = 0
        min_y = 0
        #смещает систему координат ликвидируя отрицательные координаты изображений
        for image in self.__panorama_images:
            if image.panorama_x < min_x:
                min_x = image.panorama_x
            if image.panorama_y < min_y:
                min_y = image.panorama_y
        for image in self.__panorama_images:
            image.panorama_x = int(image.panorama_x - min_x)
            image.panorama_y = int(image.panorama_y - min_y)

    def __get_panorama_shape(self):
        max_x = 0
        max_y = 0
        for image in self.__panorama_images:
            if max_x < image.panorama_x + image.image.shape[1]:
                max_x = image.panorama_x + image.image.shape[1]
            if max_y < image.panorama_y + image.image.shape[0]:
                max_y = image.panorama_y + image.image.shape[0]
        return max_x, max_y

    def __save_panorama_svg(self):
        width, height = self.__get_panorama_shape()
        dwg = sw.Drawing(size = (width, height))
        for image in self.__panorama_images:
            dwg.add(dwg.image(image.file_path, (image.panorama_x, image.panorama_y), (image.image.shape[1], image.image.shape[0])))
        dwg.saveas("panorama.svg")
        print("panorama.svg was created")

class Image:
    def __init__(self, image, key, file_path):
        self.key = key
        self.file_path = file_path
        self.image = image
        orb = cv2.ORB_create()
        self.kp, self.des = orb.detectAndCompute(self.image, None)

directory = "1"

images = Images()
for file in os.listdir(directory):
    file_path = os.path.join(directory, file)
    if os.path.isfile(file_path):
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if not image is None:
            images.addImage(file_path, image)
images.process()