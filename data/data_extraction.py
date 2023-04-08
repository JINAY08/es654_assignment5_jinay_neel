import DuckDuckGoImages as ddg
import os

def download_images(argument, folder_key):
    ddg.download(argument, max_urls = 100, folder = "data/images/" + folder_key, shuffle = True, remove_folder = True)
    path = "data/images/" + folder_key + "/"
    i = 0
    for filename in os.listdir(path):
        os.rename(os.path.join(path, filename), os.path.join(path, folder_key + str(i) + ".jpg"))
        i += 1
ddg.download('jackal', max_urls = 0, folder = "data/images/", shuffle = True, remove_folder = True)
download_images('jackal "animal"', "jackal")
download_images('nilgai "animal"', "nilgai")
