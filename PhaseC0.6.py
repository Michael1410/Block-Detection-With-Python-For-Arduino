# Detect location 2 red lego and 2 green lego, and aruco marker then send data to Arduino 

# Standard imports
import cv2
import numpy as np;
import cv2.aruco as aruco
import math
import serial
import time
from time import sleep

# Read image


cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
out = cv2.VideoWriter('c:\\outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 1)

cv2.namedWindow("TEST", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("TEST2", cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()

# Establish the connection on a specific port
ser = serial.Serial('COM3', 115200, timeout=1) 
ser.set_buffer_size(rx_size = 2147483647 , tx_size = 2147483647 )
time.sleep(1)

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
parameters = aruco.DetectorParameters_create()
print(parameters)
parameters.minDistanceToBorder = 0
parameters.adaptiveThreshWinSizeMax = 400

marker = aruco.drawMarker(aruco_dict, 200, 200)
marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

x = 0

found1 = 0
found2 = 0
rfound1 = 0
rfound2 = 0
counts = 0
countA = 0
countsB = 0
countsC = 0
# Find the distance between two coordinate points

def distance(a, b):
    d = np.linalg.norm(a - b)


while (1):
    ret, frame = cam.read()
    ret, frame2 = cam.read()
    ret, frame3 = cam.read()
    ret, frame5 = cam.read()
    if not ret:
        break
    frame4 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

    cv2.rectangle(frame4, (10, 10), (100, 100), color=(255, 0, 0), thickness=3)

    corners, ids, rejected = aruco.detectMarkers(frame4, aruco_dict, parameters=parameters)

    canvas = frame.copy()
    canvas2 = frame2.copy()
    # lower = (220,80,80)  #130,150,80
    # upper = (240,100,100) #250,250,120

    # Black
    # lower = np.array([110, 80, 20])
    # upper = np.array([140, 255, 255])

    # Green
    Grlower = np.array([0, 143, 0])
    Grupper = np.array([102, 255, 255])

    # Red
    RElower = np.array([124, 79, 40])
    REupper = np.array([180, 255, 255])

    # Green detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, Grlower, Grupper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Red Detection
    # Green detection
    Rhsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    Rmask = cv2.inRange(hsv, RElower, REupper)
    Rresult = cv2.bitwise_and(frame5, frame5, mask=Rmask)

    # Setup green SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Setup red SimpleBlobDetector parameters.
    rparams = cv2.SimpleBlobDetector_Params()

    # Big Lego Filter by Area.
    params.filterByArea = True
    params.minArea = 1000
    params.maxArea = 8000

    # Small Lego Filter by Area.
    rparams.filterByArea = True
    rparams.minArea = 1000
    rparams.maxArea = 8000

    # Filter by Circularity
    params.filterByCircularity = False
    rparams.filterByCircularity = False
    # params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    rparams.filterByConvexity = False
    # params.minConvexity = 0.1

    # Filter by Inertia
    params.filterByInertia = False
    rparams.filterByInertia = False
    # params.minInertiaRatio = 0.1

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)
    rdetector = cv2.SimpleBlobDetector_create(rparams)

    # For Green
    reversemask = 255 - mask
    blur = cv2.GaussianBlur(reversemask, (9, 9), 0)

    # For red
    Rreversemask = 255 - Rmask
    Rblur = cv2.GaussianBlur(Rreversemask, (9, 9), 0)

    # Detect green blobs.
    Grkeypoints = detector.detect(blur)

    # Detect red blobs.
    rkeypoints = rdetector.detect(Rblur)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_Grkeypoints = cv2.drawKeypoints(frame, Grkeypoints, np.array([]), (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    im_with_Grkeypoints = cv2.drawKeypoints(frame, Grkeypoints, np.array([]), (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for i in range(0, len(Grkeypoints)):
        GRx = Grkeypoints[i].pt[0]  # i is the index of the blob you want to get the position
        GRy = Grkeypoints[i].pt[1]
        GRarea = Grkeypoints[i].size
        GRxy = np.array((GRx, GRy))
        GRxy = np.array((GRx, GRy))

        # First Big Lego found coordinates________________________________________________Important
        if (i == 0 and GRarea > 45 and GRarea < 100) and counts <= 50:
            FirstLegox = Grkeypoints[i].pt[0]
            FirstLegoy = Grkeypoints[i].pt[1]
            FirstLegoxy = (FirstLegox,FirstLegoy)
            found1 = 1
            #value = float(ser.readline().strip())
            #time.sleep(0.1)

            # Send x and y coordinates to the Arduino
            ser.write("{:.2f}x\n".format(FirstLegox).encode())
            ser.write("{:.2f}y\n".format(FirstLegoy).encode())
            print(ser.readline().strip())
            counts = counts + 1

        # Second Big Lego Found Coordinates______________________________________________

            if (i == 1 and GRarea > 45 and GRarea < 100 and countA <= 50):
                SecondLegox = Grkeypoints[1].pt[0]
                SecondLegoy = Grkeypoints[1].pt[1]
                SecondLegoxy = (SecondLegox, SecondLegoy)
                found2 = 1
                #time.sleep(0.1)
                
                # Send x and y coordinates to the Arduino
                ser.write("{:.2f}q\n".format(SecondLegox).encode())
                ser.write("{:.2f}w\n".format(SecondLegoy).encode())
                print(ser.readline().strip())
                countA = countA +1
                #time.sleep(0.01)
                #print("First x: ", FirstLegox, " : Second X: ", SecondLegox)
                #print("First Y: ", FirstLegoy, " : Second Y: ", SecondLegoy)


        cv2.putText(im_with_Grkeypoints, str(int(GRx)), (int(GRx), int(GRy)), font, 1, (0, 0, 0))
        cv2.putText(im_with_Grkeypoints, str(int(GRy)), (int(GRx), int(GRy + 30)), font, 1, (0, 0, 0))
        cv2.putText(im_with_Grkeypoints, str(int(GRarea)), (int(GRx), int(GRy + 60)), font, 1, (0, 0, 0))
    # frame = cv2.bitwise_and(frame, im_with_keypoints, mask=mask)
    # cv2.imshow('frame', frame)
    # cv2.imshow('canvas',canvas)
    cv2.imshow('reversemask', reversemask)
    cv2.imshow('blur', blur)
    cv2.imshow('result', result)
    cv2.imshow('Green Keypoints', im_with_Grkeypoints)

# _________________________Begin Red Lego Detection_________________________________________________________________________________
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_rkeypoints = cv2.drawKeypoints(frame5, rkeypoints, np.array([]), (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    im_with_rkeypoints = cv2.drawKeypoints(frame5, rkeypoints, np.array([]), (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for z in range(0, len(rkeypoints)):
        Rx = rkeypoints[z].pt[0]  # i is the index of the blob you want to get the position
        Ry = rkeypoints[z].pt[1]
        Rarea = rkeypoints[z].size
        Rxy = np.array((Rx, Ry))
        Rxy = np.array((Rx, Ry))

        # First Small Lego found coordinates________________________________________________Important
        if (z == 0 and Rarea > 45 and Rarea < 100 and countsB <= 50):
            rFirstLegox = rkeypoints[z].pt[0]
            rFirstLegoy = rkeypoints[z].pt[1]
            rFirstLegoxy = (rFirstLegox, rFirstLegoy)
            rfound1 = 1
            #time.sleep(0.1)
            
            # Send x and y coordinates to the Arduino
            ser.write("{:.2f}e\n".format(rFirstLegox).encode())
            ser.write("{:.2f}r\n".format(rFirstLegoy).encode())
            print(ser.readline().strip())
            countsB = countsB + 1


        # Second Big Lego Found Coordinates______________________________________________Important
        if (z == 1 and Rarea > 45 and Rarea < 100 and countsC <= 50):
            rSecondLegox = rkeypoints[1].pt[0]
            rSecondLegoy = rkeypoints[1].pt[1]
            rSecondLegoxy = (rSecondLegox, rSecondLegoy)
            rfound2 = 1
            #time.sleep(0.1)
            
            # Send x and y coordinates to the Arduino
            ser.write("{:.2f}t\n".format(rSecondLegox).encode())
            ser.write("{:.2f}l\n".format(rSecondLegoy).encode())
            print(ser.readline().strip())
            countsC = countsC + 1

            #time.sleep(0.01)
            #print("First red lego x: ", rFirstLegox, " : Second red X: ", rSecondLegox)
            #print("First red Y: ", rFirstLegoy, " : Second red Y: ", rSecondLegoy)

        cv2.putText(im_with_rkeypoints, str(int(Rx)), (int(Rx), int(Ry)), font, 1, (0, 0, 0))
        cv2.putText(im_with_rkeypoints, str(int(Ry)), (int(Rx), int(Ry + 30)), font, 1, (0, 0, 0))
        cv2.putText(im_with_rkeypoints, str(int(Rarea)), (int(Rx), int(Ry + 60)), font, 1, (0, 0, 0))
    # frame = cv2.bitwise_and(frame, im_with_keypoints, mask=mask)
    # cv2.imshow('frame', frame)
    # cv2.imshow('canvas',canvas)
    #cv2.imshow('reversemask', Rreversemask)
    #cv2.imshow('blur', Rblur)
    cv2.imshow('result', Rresult)
    cv2.imshow('Red Keypoints', im_with_rkeypoints)
# ---------------------------End Red Lego Detector--------------------------------------------------------------------------------------------


    # Localization of the ArUco Marker for the robot________________________________________________________
    for (j, b) in enumerate(corners):
        #print(j, b, ids[j])
        #print("B0", b[0])
        #print("B00", b[0][0])
        #print("B01", b[0][1])
        #print("B02", b[0][2])

        # Corner points of the marker_________________________________________
        c1 = (b[0][0][0], b[0][0][1])
        c2 = (b[0][1][0], b[0][1][1])
        c3 = (b[0][2][0], b[0][2][1])
        c4 = (b[0][3][0], b[0][3][1])

        # Midpoint of marker__________________________________________Important
        midx = (((b[0][0][0]) + (b[0][1][0])) / 2)
        midy = (((b[0][0][1]) + (b[0][1][1])) / 2)
        Midxy = (midx, midy)
        mx = int(midx)
        my = int(midy)
        mxy = (mx, my)
        # midpoint = np.array((((b[0][0][0])+(b[0][1][0]))/2, ((b[0][0][1])+(b[0][1][1]))/2))
        midpoint = ((((b[0][0][0]) + (b[0][1][0])) / 2, ((b[0][0][1]) + (b[0][1][1])) / 2))
        ser.write("{:.2f}u\n".format(mx).encode())
        ser.write("{:.2f}i\n".format(my).encode())
        print(ser.readline().strip())
        # ----------------------------------------------------------------------------------

        # Print Corners of marker and midpoint
        #print("Midpoint of bot: ", midpoint)
        #print("Top Left: ", c1)
        #print("Top Right: ", c2)
        #print("Bottom Right: ", c3)
        #print("Bottom Left: ", c4)

        # Find Centriod of ArUco marker_____________________________________________________Important
        xCentriod = (((b[0][0][0]) + (b[0][2][0])) / 2)
        yCentriod = (((b[0][0][1]) + (b[0][2][1])) / 2)
        CentriodXY = (xCentriod, yCentriod)
        xc = int(xCentriod)
        yc = int(yCentriod)
        xyc = (xc, yc)
        #print("Centriod of bot: ", CentriodXY)
        # --------------------------------------------------------------------------------------------

        # __________________Find the angle of the robot_________________________________________________________Important
        myradiansBot = math.atan2(yCentriod - midy, xCentriod - midx)
        #print("Angle in Radians: ", myradiansBot)
        BotAngle = math.degrees(myradiansBot)
        #ser.write("{:.2f}o\n".format(BotAngle).encode())
        #print(ser.readline().strip())

        #print("Angle in Degrees: ", BotAngle)
        # ______________________________________________________________________________________
        # CentriodXY = (((b[0][0][0])+(b[0][2][0]))/2, ((b[0][0][1])+(b[0][2][1]))/2)
        # Euclidean distance between marker and block
        # dist = np.linalg.norm(midpoint - GRxy)

        # Coordinate distance between marker and the big first lego________________________________

        if (found1 == 1):

            xDistanceBetweenFirst = midx - FirstLegox
            yDistanceBetweenFirst = midy - FirstLegoy
            #print("Distance Between Bot center and First lego: ", (xDistanceBetweenFirst, yDistanceBetweenFirst))
            #print("First Lego: ", (FirstLegox, FirstLegoy))

            # Find Angle Between robot and first big lego_______________________________________________________Important
            myradiansBotBLf = math.atan2(midy - FirstLegoy, midx - FirstLegox)
            # print("Angle in Radians: ", myradiansBot)
            angleOfFirstBL = math.degrees(myradiansBotBLf)
            #ser.write("{:.2f}p\n".format(angleOfFirstBL).encode())
            #print(ser.readline().strip())
            #print("Angle between bot and first big lego: ", angleOfFirstBL)

        # Coordinate distance between marker and the Second big lego________________________________________

        if (found2 == 1):

            xDistanceBetweenSecond = midx - SecondLegox
            yDistanceBetweenSecond = midy - SecondLegoy
            #print("Distance Between Bot center and Second lego: ", (xDistanceBetweenSecond, yDistanceBetweenSecond))

            # Find Angle Between robot and second big lego________________________________Important
            myradiansBotBLs = math.atan2(midy - SecondLegoy, midx - SecondLegox)
            # print("Angle in Radians: ", myradiansBot)
            angleOfSecondBL = math.degrees(myradiansBotBLs)

            #ser.write("{:.2f}a\n".format(angleOfSecondBL).encode())
            #print(ser.readline().strip())
            #print("Angle between bot and Second big lego: ", angleOfSecondBL)
        # print((coX,coY))
        # print(dist)
        # --------------------------------------------------------------------------------------------------------

# -----------------------Red Lego Calculations------------------------------------------------------------------
            # Coordinate distance between marker and the small first lego________________________________

            if (rfound1 == 1):
                xDistanceBetweenFirstS = midx - rFirstLegox
                yDistanceBetweenFirstS = midy - rFirstLegoy
                #print("Distance Between Bot center and First lego: ", (xDistanceBetweenFirstS, yDistanceBetweenFirstS))
                #print("First Lego: ", (rFirstLegox, rFirstLegoy))

                # Find Angle Between robot and first big lego_______________________________________________________Important
                myradiansBotSf = math.atan2(midy - rFirstLegoy, midx - rFirstLegox)
                # print("Angle in Radians: ", myradiansBot)
                angleOfFirstS = math.degrees(myradiansBotSf)
                #print("Angle between bot and first small lego: ", angleOfFirstS)
                #ser.write("{:.2f}s\n".format(angleOfFirstS).encode())
                #print(ser.readline().strip())

            # Coordinate distance between marker and the Second small lego________________________________________

            if (rfound2 == 1):
                xDistanceBetweenSecondS = midx - rSecondLegox
                yDistanceBetweenSecondS = midy - rSecondLegoy
                #print("Distance Between Bot center and Second small lego: ", (xDistanceBetweenSecondS, yDistanceBetweenSecondS))

                # Find Angle Between robot and second big lego________________________________Important
                myradiansBotSs = math.atan2(midy - rSecondLegoy, midx - rSecondLegox)
                # print("Angle in Radians: ", myradiansBot)
                angleOfSecondS = math.degrees(myradiansBotSs)
                #ser.write("{:.2f}d\n".format(angleOfSecondS).encode())
                #print(ser.readline().strip())
                #print("Angle between bot and Second small lego: ", angleOfSecondS)

# -------------------------End Small Lego Calculations---------------------------------------------------------

        # Draw the lines around the marker_____________________________________________________________________
        cv2.line(frame3, c1, c2, (0, 0, 255), 3)
        cv2.line(frame3, c2, c3, (0, 0, 255), 3)
        cv2.line(frame3, c3, c4, (0, 0, 255), 3)
        cv2.line(frame3, c4, c1, (0, 0, 255), 3)
        cv2.line(frame3, xyc, mxy, (255, 0, 0), 3)
        if(found1 == 1):
            Fx1 = int(FirstLegox)
            Fy1 = int(FirstLegoy)
            Fxy1 = (Fx1, Fy1)
            cv2.line(frame3, mxy, Fxy1, (0, 155, 255), 3)
        if(found2 == 1):
            Fx2 = int(SecondLegox)
            Fy2 = int(SecondLegoy)
            Fxy2 = (Fx2, Fy2)
            cv2.line(frame3, mxy, Fxy2, (80, 155, 255), 3)

        if (rfound1 == 1):
            rFx1 = int(rFirstLegox)
            rFy1 = int(rFirstLegoy)
            rFxy1 = (rFx1, rFy1)
            cv2.line(frame3, mxy, rFxy1, (155, 0, 255), 3)
        if (rfound2 == 1):
            rFx2 = int(rSecondLegox)
            rFy2 = int(rSecondLegoy)
            rFxy2 = (rFx2, rFy2)
            cv2.line(frame3, mxy, rFxy2, (155, 0, 255), 3)
        # cv2.line(frame3, c1, c3, (0, 0, 255), 3)
        # cv2.line(frame3, c2, c4, (0, 0, 255), 3)
        # cv2.line(frame3, CentriodXY, Midxy, (0, 0, 255), 3)
        x = int((c1[0] + c2[0] + c3[0] + c4[0]) / 4)
        y = int((c1[1] + c2[1] + c3[1] + c4[1]) / 4)
        # print(b[0][0][0]-GRx)
        frame3 = cv2.putText(frame3, str(ids[j]), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (0, 0, 255), 2, cv2.LINE_AA)

        frame3 = cv2.putText(frame3, "{0},{1}".format(int(c1[0]), int(c1[1])), (c1[0], c1[1]),
                             cv2.FONT_HERSHEY_SIMPLEX,
                             0.5, (0, 0, 255), 1, cv2.LINE_AA)
        frame3 = cv2.putText(frame3, "{0},{1}".format(int(c2[0]), int(c2[1])), (c2[0], c2[1]),
                             cv2.FONT_HERSHEY_SIMPLEX,
                             0.5, (0, 0, 255), 1, cv2.LINE_AA)
        frame3 = cv2.putText(frame3, "{0},{1}".format(int(c3[0]), int(c3[1])), (c3[0], c3[1]),
                             cv2.FONT_HERSHEY_SIMPLEX,
                             0.5, (0, 0, 255), 1, cv2.LINE_AA)
        frame3 = cv2.putText(frame3, "{0},{1}".format(int(c4[0]), int(c4[1])), (c4[0], c4[1]),
                             cv2.FONT_HERSHEY_SIMPLEX,
                             0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # -----------------------------------------------------------------------------------------------

    cv2.imshow('TEST', frame3)
    cv2.imshow('TEST2', frame4)
    out.write(frame3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
