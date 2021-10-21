import cv2
import os
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import numpy as np

def ListOfImageFromPath(imageFolderPath):
    imagePathList = []
    imageList = []
    for imageName in os.listdir(imageFolderPath):
        imagePathList.append(os.path.join(imageFolderPath, imageName))
    sortedImagePathList = sorted(imagePathList, key=lambda a: int(a.split("/")[-1].split('.')[0]) )
    for path in sortedImagePathList:
        imageList.append(cv2.imread(path))
    return imageList

def make_video(outvid, images, outimg=None, fps=10, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.
 
    @param      outvid      output video
    @param      images      list of images to use in the video (not path)
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
 
    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for img in images:
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


if __name__ == '__main__':
    videoPath = '/home/saptarshi/PythonCode/AdvanceLSTM/Results/US101.mp4'
    mergeVideo = False
    if(not mergeVideo):
        singleImageList = ListOfImageFromPath('/home/saptarshi/PythonCode/AdvanceLSTM/Results/US101LaneChange')
        make_video(videoPath, singleImageList, size=(singleImageList[0].shape[1],singleImageList[0].shape[0]), format=['M', 'J', 'P', 'G'])
    else:
        ImageListUp = ListOfImageFromPath('/home/saptarshi/PythonCode/AdvanceLSTM/Results/i80200epochs/')
        ImageListDown = ListOfImageFromPath('/home/saptarshi/PythonCode/AdvanceLSTM/Results/I80LaneChnage/')
        mergedImageList = []
        for idx,_ in enumerate(ImageListUp):
            upImage = ImageListUp[idx]
            downImage = ImageListDown[idx]
            mergeImage = np.zeros((800,3600))
            mergeImage = np.concatenate((upImage,downImage), axis=0)
            mergedImageList.append(mergeImage)
        make_video(videoPath, mergedImageList, size=(mergedImageList[0].shape[1],mergedImageList[0].shape[0]), format=['M', 'J', 'P', 'G'])

  #  print(images_list)
    
