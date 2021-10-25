import os
import requests

# To check if a file exists, and download it not
def check_and_dwnld(floc, link):
    if not os.path.isfile(floc):
        if link is not None:
            print("Downloading classifier checkpoint from: " + link + " ...")
            r = requests.get(link)
            with open(floc, 'wb') as f:
                f.write(r.content)
            print("Download complete!")
        else:
            print("No model checkpoint found for this game, and none can be downloaded. \
                Add a checkpoint to the relevant environment folder to continue")
            exit()
    else:
        print("Using model found at: " + floc)
