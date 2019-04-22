from __future__ import print_function
import numpy as np
import cv2
import os
import sys
from random import randint

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]: 
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
    
  return tracker

if __name__ == '__main__':

  #print("Default tracking algoritm is CSRT \n"
  #      "Available tracking algorithms are:\n")
  #for t in trackerTypes:
  #    print(t)      
  
  trackerType = 'CSRT'      
  videoPath = sys.argv[1]
  
  # Create a video capture object to read videos 
  cap = cv2.VideoCapture(videoPath)
  # Set video to load
  success, frame123 = cap.read()
  hulk = cv2.imread('pitch.png')
  original = cv2.imread('pitch.png')
  f = open( 'file.txt', 'w' )
  # quit if unable to read the video file
  if not success:
    print('Failed to read video')
    sys.exit(1)

  ## Select boxes
  bboxes = []
  colors = [] 
  p=0
  # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
  # So we will call this function in a loop till we are done selecting all objects
  while True:
    # draw bounding boxes over objects
    # selectROI's default behaviour is to draw box starting from the center
    # when fromCenter is set to false, you can draw box starting from top left corner
    # hardcoded for pano.mp4 
    cv2.circle(frame123, (583, 50), 5, (0, 0, 255), -1)
    cv2.circle(frame123, (1342, 50), 5, (0, 0, 255), -1)
    cv2.circle(frame123, (11, 390), 5, (0, 0, 255), -1)
    cv2.circle(frame123, (1911, 390), 5, (0, 0, 255), -1)
    pts1 = np.float32([[583, 50], [1342, 50], [25, 390], [1911, 390]])
    pts2 = np.float32([[0, 0], [1800, 0], [0, 600], [1800, 600]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    frame = cv2.warpPerspective(frame123, matrix, (1800, 600))
    print('Select the 10 Home Team Outfield Players')
    print('Select the 10 Away Team Outfield Players')
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    if (p<10):
      colors.append((0,0,255))
    else:
      colors.append((255,0,0))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    p=p+1
    k = cv2.waitKey(0) & 0xFF
    if (k == 113):  # q is pressed
      break
  print('Selected bounding boxes {}'.format(bboxes))

  ## Initialize MultiTracker
  # There are two ways you can initialize multitracker
  # 1. tracker = cv2.MultiTracker("CSRT")
  # All the trackers added to this multitracker
  # will use CSRT algorithm as default
  # 2. tracker = cv2.MultiTracker()
  # No default algorithm specified

  # Initialize MultiTracker with tracking algo
  # Specify tracker type
  
  # Create MultiTracker object
  multiTracker = cv2.MultiTracker_create()

  # Initialize MultiTracker 
  for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)


  # Process video and track objects
  while cap.isOpened():
    success, frame123 = cap.read()
    # hardcoded for pano.mp4 
    cv2.circle(frame123, (583, 50), 5, (0, 0, 255), -1)
    cv2.circle(frame123, (1342, 50), 5, (0, 0, 255), -1)
    cv2.circle(frame123, (11, 390), 5, (0, 0, 255), -1)
    cv2.circle(frame123, (1911, 390), 5, (0, 0, 255), -1)
    pts1 = np.float32([[583, 50], [1342, 50], [25, 390], [1911, 390]])
    pts2 = np.float32([[0, 0], [1800, 0], [0, 600], [1800, 600]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    frame = cv2.warpPerspective(frame123, matrix, (1800, 600))
    if not success:
      break
    
    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)
    l,b ,channels = frame.shape  #newaddition
    hulk = cv2.resize(hulk,(b,l))
    original = cv2.resize(original,(b,l))
    radar = original.copy()
    # draw tracked objects
    for i, newbox in enumerate(boxes):
      p1 = (int(newbox[0]), int(newbox[1]))
      p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
      cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

      if (i<10):
        cv2.putText(frame, str(i), (int(newbox[0]), int(newbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
        #cv2.putText(hulk, 'Home', p1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)#newaddition
        #cv2.putText(hulk, str(i), (int(newbox[0]), int(newbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)  
        overlay=hulk.copy()
        cv2.circle(overlay,(int(newbox[0]), int(newbox[1])), 10, (0,0,255), -1)#newaddition
        #cv2.applyColorMap(hulk,cv2.COLORMAP_AUTUMN)
        cv2.circle(radar,(int(newbox[0]), int(newbox[1])), 10, (0,0,255), -1)#newaddition
        alpha = 0.1  # Transparency factor.

        # Following line overlays transparent rectangle over the image
        hulk = cv2.addWeighted(overlay, alpha, hulk, 1 - alpha, 0)

        f.write( 'Home: Player '+str(i)+' x,y: '+str(int(newbox[0]))+','+str(int(newbox[1])) + '\n' )   #####
      else:
        cv2.putText(frame, str(i), (int(newbox[0])-2, int(newbox[1])-2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1, cv2.LINE_AA)
        #cv2.putText(hulk, 'Away', p1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)#newaddition
        #cv2.putText(hulk, str(i), (int(newbox[0]), int(newbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1, cv2.LINE_AA)  
        overlay=hulk.copy()
        cv2.circle(overlay,(int(newbox[0]), int(newbox[1])), 10, (255,0,0), -1)#newaddition
        #cv2.applyColorMap(hulk,cv2.COLORMAP_AUTUMN)
        cv2.circle(radar,(int(newbox[0]), int(newbox[1])), 10, (255,0,0), -1)#newaddition
        alpha = 0.1  # Transparency factor.

        # Following line overlays transparent rectangle over the image
        hulk = cv2.addWeighted(overlay, alpha, hulk, 1 - alpha, 0)
        f.write( 'Away: Player '+str(i)+' x,y: '+str(int(newbox[0]))+','+str(int(newbox[1])) + '\n' )   #####
    # show frame
    cv2.imshow('MultiTracker', frame)
    cv2.imshow('HeatMap',hulk) #newaddition
    cv2.imshow('Radar',radar)
    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
f.close()



