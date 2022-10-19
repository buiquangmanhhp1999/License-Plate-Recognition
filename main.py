import cv2
from pathlib import Path
import argparse
import time
from PIL import Image

from src.lp_recognition import E2E



def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='./samples/1.jpg')

    return arg.parse_args()


args = get_arguments()
img_path = Path(args.image_path)

# read image
img = cv2.imread(str(img_path))

# start
start = time.time()

# load model
model = E2E()

# recognize license plate
image = model.predict(img)

# end
end = time.time()

print('Model process on %.2f s' % (end - start))

# show image
cv2.imshow('License Plate', image)

# save result
# Author: HuyPham 15/9/2022
# https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
im = Image.fromarray(image)
im.save("result.jpeg")
#end

if cv2.waitKey(0) & 0xFF == ord('q'):
    exit(0)


cv2.destroyAllWindows()
