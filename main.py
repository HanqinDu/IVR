#!/usr/bin/env python2.7
import gym
import reacher3D.Reacher
import numpy as np
import cv2
import math
import scipy as sp
import collections
import time
class MainReacher():
    def __init__(self):
        self.env = gym.make('3DReacherMy-v0')
        self.env.reset()

    def coordinate_convert_3D(self,pixels):
        #Converts pixels into metres
        return np.array([(pixels[0]-self.env.viewerSize/2)/self.env.resolution,-(pixels[1]-self.env.viewerSize/2)/self.env.resolution,-(pixels[2]-self.env.viewerSize/2)/self.env.resolution])

    def coordinate_convert(self,pixels):
        #Converts pixels into metres
        return np.array([(pixels[0]-self.env.viewerSize/2)/self.env.resolution,-(pixels[1]-self.env.viewerSize/2)/self.env.resolution])

    def detect_green(self,image_xy,image_xz):
        #In this method you should focus on detecting the center of the green circle
        #SAME AS DETECT_BLUE JUST WITH DIFFERENT COLOUR LIMITS
        mask_xy = cv2.inRange(image_xy, (55,5,0),(65,255,255))
        mask_xz = cv2.inRange(image_xy, (55,5,0),(65,255,255))
        kernel = np.ones((5,5),np.uint8)
        mask_xy = cv2.dilate(mask_xy,kernel,iterations=3)
        mask_xz = cv2.dilate(mask_xz,kernel,iterations=3)
        M = cv2.moments(mask_xy)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        M = cv2.moments(mask_xz)
        cx = (int(M['m10']/M['m00']) + cx)/2
        cz = int(M['m01']/M['m00'])

        return self.coordinate_convert_3D(np.array([cx,cy,cz]))

    def detect_red(self,image_xy,image_xz):
        #In this method you should focus on detecting the center of the green circle
        #SAME AS DETECT_BLUE JUST WITH DIFFERENT COLOUR LIMITS

        mask_xy = cv2.inRange(image_xy, (115,5,0),(125,255,255))
        mask_xz = cv2.inRange(image_xy, (115,5,0),(125,255,255))
        kernel = np.ones((5,5),np.uint8)
        mask_xy = cv2.dilate(mask_xy,kernel,iterations=3)
        mask_xz = cv2.dilate(mask_xz,kernel,iterations=3)
        M = cv2.moments(mask_xy)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        M = cv2.moments(mask_xz)
        cx = (int(M['m10']/M['m00']) + cx)/2
        cz = int(M['m01']/M['m00'])

        return self.coordinate_convert_3D(np.array([cx,cy,cz]))

    def detect_blue(self,image_xy,image_xz,Vpeak):
        #In this method you should focus on detecting the center of the green circle
        #SAME AS DETECT_BLUE JUST WITH DIFFERENT COLOUR LIMITS
        mask_xy = cv2.inRange(image_xy, (0,5,(Vpeak*3/4)),(5,255,255))
        #mask_xy = mask_xy + cv2.inRange(image_xy, (170,0,0),(180,255,255))
        mask_xz = cv2.inRange(image_xy, (0,5,(Vpeak*3/4)),(5,255,255))
        #mask_xz = mask_xz + cv2.inRange(image_xy, (170,0,0),(180,255,255))
        kernel = np.ones((5,5),np.uint8)
        mask_xy = cv2.dilate(mask_xy,kernel,iterations=3)
        mask_xz = cv2.dilate(mask_xz,kernel,iterations=3)
        M = cv2.moments(mask_xy)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        M = cv2.moments(mask_xz)
        cx = (int(M['m10']/M['m00']) + cx)/2
        cz = int(M['m01']/M['m00'])

        return self.coordinate_convert_3D(np.array([cx,cy,cz]))

    def detect_dblue(self,image_xy,image_xz,Vpeak):
        #In this method you should focus on detecting the center of the green circle
        #SAME AS DETECT_BLUE JUST WITH DIFFERENT COLOUR LIMITS
        mask_xy = cv2.inRange(image_xy, (0,5,0),(5,255,(Vpeak*3/4)))
        #mask_xy = mask_xy + cv2.inRange(image_xy, (170,0,0),(180,255,255))
        mask_xz = cv2.inRange(image_xy, (0,5,0),(5,255,(Vpeak*3/4)))
        #mask_xz = mask_xz + cv2.inRange(image_xy, (170,0,0),(180,255,255))
        kernel = np.ones((5,5),np.uint8)
        mask_xy = cv2.dilate(mask_xy,kernel,iterations=3)
        mask_xz = cv2.dilate(mask_xz,kernel,iterations=3)
        M = cv2.moments(mask_xy)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        M = cv2.moments(mask_xz)
        cx = (int(M['m10']/M['m00']) + cx)/2
        cz = int(M['m01']/M['m00'])

        return self.coordinate_convert_3D(np.array([cx,cy,cz]))



    def go(self):
        #The robot has several simulated modes:
        #These modes are listed in the following format:
        #Identifier (control mode) : Description : Input structure into step function

        #POS : A joint space position control mode that allows you to set the desired joint angles and will position the robot to these angles : env.step((np.zeros(3),np.zeros(3), desired joint angles, np.zeros(3)))
        #POS-IMG : Same control as POS, however you must provide the current joint angles and velocities : env.step((estimated joint angles, estimated joint velocities, desired joint angles, np.zeros(3)))
        #VEL : A joint space velocity control, the inputs require the joint angle error and joint velocities : env.step((joint angle error (velocity), estimated joint velocities, np.zeros(3), np.zeros(3)))
        #TORQUE : Provides direct access to the torque control on the robot : env.step((np.zeros(3),np.zeros(3),np.zeros(3),desired joint torques))
        self.env.controlMode="POS"
        #Run 100000 iterations
        prev_JAs = np.zeros(4)
        prev_jvs = collections.deque(np.zeros(4),1)

        # Uncomment to have gravity act in the z-axis
        # self.env.world.setGravity((0,0,-9.81))

        # test
        thresholds=np.zeros([3])

        for loop in range(100000):
            #The change in time between iterations can be found in the self.env.dt variable
            dt = self.env.dt
            #self.env.render returns 2 RGB arrays of the robot, one for the xy-plane, and one for the xz-plane
            arrxy,arrxz = self.env.render('rgb-array')


            # D: calculate joint position
            # Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].

            arrxy_HSV = cv2.cvtColor(arrxy, cv2.COLOR_BGR2HSV)
            arrxz_HSV = cv2.cvtColor(arrxz, cv2.COLOR_BGR2HSV)

            img_hist=cv2.calcHist([arrxy],[2],None,[256],[0,256])

            Vpeak = np.argmax(img_hist)

            jointPos1 = self.detect_red(arrxy_HSV,arrxz_HSV)
            jointPos2 = self.detect_green(arrxy_HSV,arrxz_HSV)
            jointPos3 = self.detect_blue(arrxy_HSV,arrxz_HSV,Vpeak)
            jointPos4 = self.detect_dblue(arrxy_HSV,arrxz_HSV,Vpeak)


            # test

            #cv2.imwrite("arrxy.jpg",arrxy)

            # Etest

            jointAngles = np.array([0.5,0.5,0.5,-0.5])

            self.env.step((np.zeros(4),np.zeros(4),jointAngles, np.zeros(4)))
            #self.env.step((np.zeros(4),np.zeros(4),np.zeros(4), np.zeros(4)))
            #The step method will send the control input to the robot, the parameters are as follows: (Current Joint Angles/Error, Current Joint Velocities, Desired Joint Angles, Torque input)

#main method
def main():
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()
