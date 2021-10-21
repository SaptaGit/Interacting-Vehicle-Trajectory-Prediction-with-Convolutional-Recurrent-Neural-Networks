
# import os
# import shutil

# trainFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/small/'
# trainList = sorted(os.listdir(trainFolder), key=int)

# for name in trainList:
#     if (int(name) > 140):
#         path = trainFolder + name
#         shutil.rmtree(path)

import cv2
import numpy as np

image = cv2.imread('/home/saptarshi/PythonCode/AdvanceLSTMServer/TrainedModels/3691.png', cv2.IMREAD_COLOR).astype('float16')
print('Loaded...')