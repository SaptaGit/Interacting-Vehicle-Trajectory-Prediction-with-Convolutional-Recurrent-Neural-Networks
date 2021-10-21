# Extract the labbeled bounding boxes and create OGMs from training.
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import math
# import pptk

sequence_folder = '/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/junction_1'
globalRadarDisplayImage = np.zeros((1152,1152,3))
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
turn = 0
straight = 0
processedObjectIds = []

class Sequence:

    def __init__(self, sequence_path):
        self.sequence_path = sequence_path

    def draw_boundingbox_rot(self, im, bbox, angle, color):
        theta = np.deg2rad(-angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        points = np.array([[bbox[0], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                           [bbox[0], bbox[1] + bbox[3]]]).T

        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        T = np.array([[cx], [cy]])

        points = points - T
        points = np.matmul(R, points) + T
        points = points.astype(int)

        #color = (255, 0, 0)
        color = np.array(color) * 255

        cv2.line(im, tuple(points[:, 0]), tuple(points[:, 1]), color, 3)
        cv2.line(im, tuple(points[:, 1]), tuple(points[:, 2]), color, 3)
        cv2.line(im, tuple(points[:, 2]), tuple(points[:, 3]), color, 3)
        cv2.line(im, tuple(points[:, 3]), tuple(points[:, 0]), color, 3)

        return im

    def load_annotations(self, annotation_path):
        if (os.path.exists(annotation_path)):
            f = open(annotation_path)
            self.annotations = json.load(f)
        else:
            self.annotations = None

    def load_timestamp(self, timestamp_path):
        genfromtxt = np.genfromtxt(
            timestamp_path, dtype=(str, int, str, float))
        timestamps = {'frame': [], 'time': []}
        for line in genfromtxt:
            timestamps['frame'].append(line[1])
            timestamps['time'].append(line[3])
        return timestamps

    # get the ids from radar ids
    def get_frame_ids(self, timestamps_radar,):

        self.total_nb_frames_radar = len(timestamps_radar['frame'])
        radar_ids = np.array(np.arange(1,self.total_nb_frames_radar))
        time = timestamps_radar['time']
        return radar_ids

    def load_sequence(self,sequence_path):
        # get all time stamps
        timestamps_radar_path = os.path.join(sequence_path, 'Navtech_Cartesian.txt')
        timestamps_radar = self.load_timestamp(timestamps_radar_path)

        radar_ids = self.get_frame_ids(timestamps_radar)
        self.total_nb_frames_radar = len(timestamps_radar['frame'])

        self.radar_ids = radar_ids

    def play(self,sequence_path,idOffset):

        global globalRadarDisplayImage
        global turn
        global straight
        # v = pptk.viewer(np.array([0, 0, 0]))
        # for all radar frames

        for i in range(1, self.total_nb_frames_radar - 1):
            # get correct frames
            radar_id = int(self.radar_ids[i])

            str_format = '{:06d}'

            radar_cartesian_path = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(radar_id) + '.png')

            radar_cartesian = cv2.imread(radar_cartesian_path)
            if(i == 1 and idOffset == 0):
                globalRadarDisplayImage = cv2.imread(radar_cartesian_path) #radar_cartesian

            if (self.annotations != None):
                for object in self.annotations:
                    if (object['bboxes'][i]):
                        if (object['deleted'][i] == 0):
                            if (object['visible'][i] == 'visible'):
                                bbox = object['bboxes'][i]['position']
                                angle = object['bboxes'][i]['rotation']
                                color = object['color']
                                radar_cartesian = self.draw_boundingbox_rot(radar_cartesian, bbox, angle, color)
                                cx = int(bbox[0] + bbox[2]/2)
                                cy = int(bbox[1] + bbox[3]/2)
                                radar_cartesian = cv2.putText(radar_cartesian, str(object['id']+idOffset), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA) 
                                #radar_cartesian = cv2.putText(radar_cartesian, str(cx), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA) 
                                #check if previous frame exists
                                if(len(object['bboxes'][i-1])>0):
                                    prevBox = object['bboxes'][i-1]['position']
                                    prevCx = int(prevBox[0] + prevBox[2]/2)
                                    prevCy = int(prevBox[1] + prevBox[3]/2)
                                    # check if turn or straight trajectory. Travers the list from the back and pick the last position.
                                    # If the position is > 580 its a turn
                                    lastLocation = 0
                                    turnBool = False
                                    for j in reversed(object['bboxes']):
                                        if isinstance(j, dict):
                                            lastLocation = j['position'][0]
                                            break

                                    if (lastLocation>580):
                                        turnBool = True
                                    #globalRadarDisplayImage = cv2.circle(globalRadarDisplayImage, (cx,cy), 2, (0,0,255)) 
                                    if(turnBool):
                                        globalRadarDisplayImage = cv2.line(globalRadarDisplayImage, (cx,cy), (prevCx,prevCy), (0,0,255), 2)
                                        if object['id'] not in processedObjectIds:
                                            processedObjectIds.append(object['id'])
                                            turn = turn + 1
                                    else:
                                        globalRadarDisplayImage = cv2.line(globalRadarDisplayImage, (cx,cy), (prevCx,prevCy), (0,255,0), 2)
                                        if object['id'] not in processedObjectIds:
                                            processedObjectIds.append(object['id'])
                                            straight = straight + 1
            
            turnStr = 'Turn : ' + str (turn)
            straightStr = 'Straight : ' + str (straight)
            radar_cartesian = cv2.putText(radar_cartesian, turnStr, (800,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA) 
            radar_cartesian = cv2.putText(radar_cartesian, straightStr, (800,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA) 

            finalImage = np.hstack((radar_cartesian,globalRadarDisplayImage))

            cv2.imshow('image', finalImage)
            cv2.waitKey(1)

def __main__():

    folderList = os.listdir(sequence_folder)
    folderList.sort(key=lambda x: int(x.split('_')[-1]))
    idOffset = 0

    for eachSeq in folderList:
        sequence_path = os.path.join(sequence_folder, eachSeq)
        annotation_path = os.path.join(sequence_path, 'annotations', 'annotations.json')

        sequence = Sequence(sequence_path)
        sequence.load_sequence(sequence_path)
        sequence.load_annotations(annotation_path)
        sequence.play(sequence_path,idOffset)
        idOffset = idOffset + sorted([d['id'] for d in sequence.annotations])[-1]


__main__()
