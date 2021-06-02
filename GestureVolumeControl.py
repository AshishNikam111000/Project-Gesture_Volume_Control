import cv2, time, math
import numpy as np
import mediapipe as mp
import HandModule as htm

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


###############################
wCam, hCam = 640, 480
###############################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
ptime = 0

detector = htm.HandDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()

minVol, maxVol = volRange[0], volRange[1]
volBar = 400

while True:
    success, img = cap.read()

    img = detector.FindHands(img)
    lmlist = detector.FindPos(img, draw=False)
    if len(lmlist) !=0:
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x2 + x1)//2, (y2 + y1)//2

        # length = int(math.sqrt( (x2-x1)**2 + (y2-y1)**2 )) one way to find length
        length = int(math.hypot(x2-x1, y2-y1))

        cv2.circle(img, (cx, cy), 4, (255, 0, 255), 4)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        vol = int(np.interp(length, [50, 200], [minVol, maxVol]))
        volBar = int(np.interp(length, [50, 200], [400, 150]))
        print(length, vol)

        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv2.circle(img, (cx, cy), 4, (0, 255, 0), 4)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, volBar), (85, 400), (0, 255, 0), cv2.FILLED)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break