import DuckDuckGoImages as ddg
import os
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import PIL

def download_images(argument, folder_key):
    ddg.download(argument, max_urls = 130, folder = "data/images/", shuffle = True, safe_search = True)
    path = "data/images/" 
    i = 0
    for filename in os.listdir(path):
        try:
            photo = load_img(path + filename)
            i += 1
            os.rename(os.path.join(path, filename), os.path.join(path, folder_key + str(i) + ".jpg"))      
        except PIL.UnidentifiedImageError:
            continue
        if(i==100): break
    print(i)

ddg.download("jackal", max_urls = 0, folder = "data/images/", shuffle = True, remove_folder = True)
download_images('jackal "animal"', "jackal")
download_images('nilgai "animal"', "nilgai")


# Clean up the data folder
path = "data/images/"
for filename in os.listdir(path):
    if(filename.startswith("jackal") or filename.startswith("nilgai")):
        continue
    else:
        os.remove(os.path.join(path, filename))






