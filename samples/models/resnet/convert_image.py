from urllib.request import urlopen
from PIL import Image
import numpy as np, io

url = "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"
data = urlopen(url).read()
img = Image.open(io.BytesIO(data)).convert("RGB")
img = img.resize((224, 224), Image.BILINEAR)
arr = np.array(img).astype(np.float32)
arr = arr.reshape((1,)+arr.shape)   # (1,224,224,3)
print("writing resnet_input_1_224_224_3_f32.bin", arr.shape, arr.dtype)
arr.tofile("resnet_input_1_224_224_3_f32.bin")