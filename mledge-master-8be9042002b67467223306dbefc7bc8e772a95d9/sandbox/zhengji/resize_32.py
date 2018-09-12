import cv2
import os

dir_name = '/Data/imagenet-10'
dir_list = os.listdir(dir_name)

dest_name = '/Data/imagenet-10-32x32'
index = 0
for item in dir_list:
    class_path = os.path.join(dir_name, item)

    dest_path = os.path.join(dest_name, item)

    if os.path.isdir(class_path):
        if not os.path.exists(dest_path):
            os.mkdir(dest_path)
        image_list = os.listdir(class_path)
        for image in image_list:
            index += 1
            image_path = os.path.join(class_path, image)
            output_path = os.path.join(dest_path, image)
            img = cv2.imread(image_path)
            n, m = img.shape[0], img.shape[1]
            if n > m:
                t = (n - m) / 2
                img = img[t:t+m, :, :]
            else:
                t = (m - n) / 2
                img = img[:, t:t+n, :]
            img = cv2.resize(img, (32, 32))
            cv2.imwrite(output_path, img)
            if index % 1000 == 0:
                print index
print index
