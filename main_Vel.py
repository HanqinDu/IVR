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

    #By Du
    def coordinate_convert_3D(self,pixels):
        #Converts pixels into metres
        return np.array([(pixels[0]-self.env.viewerSize/2)/self.env.resolution,-(pixels[1]-self.env.viewerSize/2)/self.env.resolution,-(pixels[2]-self.env.viewerSize/2)/self.env.resolution])

    def coordinate_convert(self,pixels):
        #Converts pixels into metres
        return np.array([(pixels[0]-self.env.viewerSize/2)/self.env.resolution,-(pixels[1]-self.env.viewerSize/2)/self.env.resolution])

    def angle_(self, v1, v2):
        safe = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if(safe > 1):
            safe = 1
        if(safe<-1):
            safe = -1

        return (np.arccos(safe))
        #output = (np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

    def ceil(self, v, c):
        if(v>c):
            return v
        else:
            return v

    def detect_green(self,image_xy,image_xz):
        #In this method you should focus on detecting the center of the green circle
        #SAME AS DETECT_BLUE JUST WITH DIFFERENT COLOUR LIMITS
        mask_xy = cv2.inRange(image_xy, (55,5,0),(65,255,255))
        mask_xz = cv2.inRange(image_xz, (55,5,0),(65,255,255))
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
        mask_xz = cv2.inRange(image_xz, (115,5,0),(125,255,255))
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
        mask_xz = cv2.inRange(image_xz, (0,5,(Vpeak*3/4)),(5,255,255))
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

    def detect_ee(self,image_xy,image_xz,Vpeak):
        #In this method you should focus on detecting the center of the green circle
        #SAME AS DETECT_BLUE JUST WITH DIFFERENT COLOUR LIMITS
        mask_xy = cv2.inRange(image_xy, (0,5,0),(5,255,(Vpeak*3/4)))
        #mask_xy = mask_xy + cv2.inRange(image_xy, (170,0,0),(180,255,255))
        mask_xz = cv2.inRange(image_xz, (0,5,0),(5,255,(Vpeak*3/4)))
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

    def detect_target(self,image_xy,image_xz,Vpeak):
        mask_xy = cv2.inRange(image_xy, (0,0,(Vpeak*65/100)),(180,5,(Vpeak*75/100)))
        mask_xz = cv2.inRange(image_xz, (0,0,(Vpeak*65/100)),(180,5,(Vpeak*75/100)))

        imgXY,countour_XY, _ = cv2.findContours(mask_xy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgXZ,countour_XZ, _ = cv2.findContours(mask_xz, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cxy1 = np.reshape(np.mean(countour_XY[0],0,dtype = np.int),2)
        cxy2 = np.reshape(np.mean(countour_XY[1],0,dtype = np.int),2)
        cxz1 = np.reshape(np.mean(countour_XZ[0],0,dtype = np.int),2)
        cxz2 = np.reshape(np.mean(countour_XZ[1],0,dtype = np.int),2)


        for i in range(10):
            for j in range(10):
                if(mask_xy[cxy1[1]+i-5][cxy1[0]+j-5] != mask_xy[cxy2[1]+i-5][cxy2[0]+j-5]):
                    if(mask_xy[cxy1[1]+i-5][cxy1[0]+j-5] < mask_xy[cxy2[1]+i-5][cxy2[0]+j-5]):
                        target_x = cxy1[0]
                        target_y = cxy1[1]
                        i = 10
                    else:
                        target_x = cxy2[0]
                        target_y = cxy2[1]
                        j = 10

        for i in range(10):
            for j in range(10):
                if(mask_xz[cxz1[1]+i-5][cxz1[0]+j-5] != mask_xz[cxz2[1]+i-5][cxz2[0]+j-5]):
                    if(mask_xz[cxz1[1]+i-5][cxz1[0]+j-5] < mask_xz[cxz2[1]+i-5][cxz2[0]+j-5]):
                        target_x = (cxz1[0] + target_x)/2
                        target_z = cxz1[1]
                        i = 10
                    else:
                        target_x = (cxz2[0] + target_x)/2
                        target_z = cxz2[1]
                        j = 10

        return self.coordinate_convert_3D(np.array([target_x,target_y,target_z]))

    def FK(self,joint_angles):
        #Forward Kinematics to calculate end effector location
        #Each link is 1m long
        #calculate each individual jacobian transform
        j1_transform = self.link_transform_y(joint_angles[0])
        j2_transform = self.link_transform_z(joint_angles[1])
        j3_transform = self.link_transform_z(joint_angles[2])
        j4_transform = self.link_transform_y(joint_angles[3])
        #combine the transforms for each
        total_transform = j1_transform*j2_transform*j3_transform*j4_transform
        return total_transform

    def IK(self, current_joint_angles, desired_position):
        #Calculate current position and error in position in task space
        curr_pos = self.FK(current_joint_angles)[0:3,3]
        pos_error = desired_position - np.squeeze(np.array(curr_pos.T))
        #Calculate Jacobian
        Jac = np.matrix(self.Jacobian(current_joint_angles))[0:3,:]
        #Jac_inv = Jac.T
        #If the Jacobian is low rank (<2) then use the transpose otherwise use psuedo-inverse
        if(np.linalg.matrix_rank(Jac,0.4)<3):
            Jac_inv = Jac.T
        else:
            Jac_inv = Jac.T*np.linalg.inv(Jac*Jac.T)
        #Apply inverted jacobian on the position error in task space to get qdot
        q_dot = Jac_inv*np.matrix(pos_error).T
        return np.squeeze(np.array(q_dot.T))

    def ts_pd_control(self, curr_ee_pos, curr_ee_vel, desired_ee_pos):
        # Gain selection
        P = np.array([200,200,200])
        D = np.array([80,80,80])
        # Proportional error
        P_error = np.matrix(desired_ee_pos - curr_ee_pos).T
        # Damping error
        D_error = np.zeros(shape=(3,1)) - np.matrix(curr_ee_vel).T
        # PD error
        PD_error = np.diag(P)*P_error + np.diag(D)*D_error

        return PD_error

    def js_pd_control(self, current_joint_angles, current_joint_velocities, desired_joint_angles):
        # Gain selection
        P = np.array([150,150,150,150])
        D = np.array([55,35,15,5])
        # Proportional error
        P_error = np.matrix(desired_joint_angles - current_joint_angles).T
        # Damping error
        D_error = np.matrix(np.zeros(4) - current_joint_velocities).T
        # PD error
        PD_error = np.diag(P)*P_error + np.diag(D)*D_error
        return PD_error

    def link_transform_z(self,angle):
        #Calculate the Homogenoeous transformation matrix from rotation and translation
        return np.matrix([[np.cos(angle),-np.sin(angle),0,np.cos(angle)],
                          [np.sin(angle), np.cos(angle),0,np.sin(angle)],
                          [0,0,1,0],
                          [0,0,0,1]])

    def link_transform_y(self,angle):
        #Calculate the Homogenoeous transformation matrix from rotation and translation
        return np.matrix([[np.cos(angle),0,-np.sin(angle),np.cos(angle)],
                          [0,1,0,0],
                          [np.sin(angle),0,np.cos(angle),np.sin(angle)],
                          [0,0,0,1]])

    def Jacobian(self,joint_angles):
        #Forward Kinematics to calculate end effector location
        #Each link is 1m long
        #initialize matrix for jacobian
        jacobian = np.zeros((4,4))
        #calculate each individual jacobian transform
        j1_transform = self.link_transform_y(joint_angles[0])
        j2_transform = self.link_transform_z(joint_angles[1])
        j3_transform = self.link_transform_z(joint_angles[2])
        j4_transform = self.link_transform_y(joint_angles[3])
        #combine the transforms for each
        total_transform = j1_transform*j2_transform*j3_transform*j4_transform
        #obtain end effector cartesian location

        ee_pos = total_transform[0:3,3]


        #obtain joint 4 cartesian location
        j4_pos = (j1_transform*j2_transform*j3_transform)[0:3,3]
        #obtain joint 3 cartesian location
        j3_pos = (j1_transform*j2_transform)[0:3,3]
        #obtain joint 2 cartesian location
        j2_pos = (j1_transform)[0:3,3]
        #obtain joint 1 cartesian location
        j1_pos = np.zeros((3,1))
        #Initialize vector containing axis of rotation, for planar robots rotating around z this is constant
        z_vector = np.array([0,0,1])
        y_vector = np.array([0,-1,0])
        #Calculate geometric jacobian using equations
        pos_3D = np.zeros(3)

        pos_3D[0:3] = (ee_pos-j1_pos).T
        jacobian[0:3,0] = np.cross(y_vector,pos_3D)[0:3]
        z_vector = z_vector*self.rotation_matrix_y(joint_angles[0])

        pos_3D[0:3] = (ee_pos-j2_pos).T
        jacobian[0:3,1] = np.cross(z_vector,pos_3D)[0:3]
        pos_3D[0:3] = (ee_pos-j3_pos).T
        jacobian[0:3,2] = np.cross(z_vector,pos_3D)[0:3]

        y_vector = y_vector*self.rotation_matrix_w(-joint_angles[2]-joint_angles[1],z_vector.T)

        pos_3D[0:3] = (ee_pos-j4_pos).T
        jacobian[0:3,3] = np.cross(y_vector,pos_3D)[0:3]
        jacobian[3,:] = 1

        return jacobian

    def rotation_matrix_z(self, angle):
        return np.matrix([[np.cos(angle),-np.sin(angle),0],
                          [np.sin(angle),np.cos(angle),0],
                          [0,0,1]])

    def rotation_matrix_x(self, angle):
        return np.matrix([[1,0,0],
                          [0,np.cos(angle),-np.sin(angle)],
                          [0,np.sin(angle),np.cos(angle)]])

    def rotation_matrix_y(self, angle):
        return np.matrix([[np.cos(angle),0,np.sin(angle)],
                          [0,1,0],
                          [-np.sin(angle),0,np.cos(angle)]])

    def antisymmetric_matrix(self,w):
        return np.matrix([[0,-w[2],w[1]],
                          [w[2],0,-w[0]],
                          [-w[1],w[0],0]])

    def rotation_matrix_w(self, angle, w):
        w_ = self.antisymmetric_matrix(w)
        return (np.eye(3) + w_*np.sin(angle) + w_*w_*(1-np.cos(angle)))


    #By Du

    #By eris
    def detect_joint_angles(self,image_xy,image_xz,Vpeak):
        #Calculate the relevant joint angles from the image
        #Obtain the center of each coloured blob(red green blue dblue)
        jointPos1 = (self.detect_red(image_xy,image_xz))
        jointPos2 = (self.detect_green(image_xy,image_xz))
        jointPos3 = (self.detect_blue(image_xy,image_xz,Vpeak))
        jointPos4 = (self.detect_ee(image_xy,image_xz,Vpeak))

        ja1 = self.angle_(jointPos1,np.array([1,0,0]))
        ja2 = self.angle_(jointPos2-jointPos1,jointPos1)
        ja3 = self.angle_(jointPos3-jointPos2,jointPos2-jointPos1)
        ja4 = self.angle_(jointPos4-jointPos3,jointPos3-jointPos2)

        #print(jointPos4)
        #Solve using trigonometry

        z_vector = np.array([0,0,1])
        y_vector = np.array([0,-1,0])

        if((np.cross([1,0,0],jointPos1)*y_vector)[1] < 0):
            ja1 = -ja1

        z_vector = np.array(z_vector*self.rotation_matrix_y(ja1)).T

        if((np.dot(np.cross(jointPos1,jointPos2-jointPos1),z_vector)) < 0):
            ja2 = -ja2

        if((np.dot(np.cross(jointPos2,jointPos3-jointPos2),z_vector)) < 0):
            ja3 = -ja3

        y_vector = y_vector*self.rotation_matrix_w(-ja2-ja3,z_vector)

        if((np.dot(np.cross(jointPos3,jointPos4-jointPos3),np.array([y_vector[0,0],y_vector[0,1],y_vector[0,2]]))) < 0):
            ja4 = -ja4

        return np.array([ja1,ja2,ja3,ja4])

    def angle_normalize(self,x):
        #Normalizes the angle between pi and -pi
        return (((x+np.pi) % (2*np.pi)) - np.pi)
    #By eris



    def go(self):
        #The robot has several simulated modes:
        #These modes are listed in the following format:
        #Identifier (control mode) : Description : Input structure into step function

        #POS : A joint space position control mode that allows you to set the desired joint angles and will position the robot to these angles : env.step((np.zeros(3),np.zeros(3), desired joint angles, np.zeros(3)))
        #POS-IMG : Same control as POS, however you must provide the current joint angles and velocities : env.step((estimated joint angles, estimated joint velocities, desired joint angles, np.zeros(3)))
        #VEL : A joint space velocity control, the inputs require the joint angle error and joint velocities : env.step((joint angle error (velocity), estimated joint velocities, np.zeros(3), np.zeros(3)))
        #TORQUE : Provides direct access to the torque control on the robot : env.step((np.zeros(3),np.zeros(3),np.zeros(3),desired joint torques))
        self.env.controlMode="TORQUE"
        #Run 100000 iterations
        prev_JAs = np.zeros(4)
        prev_jvs = collections.deque(np.zeros(4),1)

        # Uncomment to have gravity act in the z-axis
        # self.env.world.setGravity((0,0,-9.81))

        # test
        prevEePos = np.zeros(shape=(3,1))
        ee_target_pre = np.zeros(shape=(3,1))

        timeused = 0

        switch = False

        for loop in range(100000):

            #self.env.render returns 2 RGB arrays of the robot, one for the xy-plane, and one for the xz-plane
            arrxy,arrxz = self.env.render('rgb-array')

            dt = self.env.dt

            # D: calculate joint position
            # Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].

            arrxy_HSV = cv2.cvtColor(arrxy, cv2.COLOR_BGR2HSV)
            arrxz_HSV = cv2.cvtColor(arrxz, cv2.COLOR_BGR2HSV)

            img_hist=cv2.calcHist([arrxy],[2],None,[256],[0,256])
            Vpeak = np.argmax(img_hist)

            detectedJointAngles = self.detect_joint_angles(arrxy_HSV,arrxz_HSV,Vpeak)

            #velocity part [eris]
            #The change in time between iterations can be found in the self.env.dt variable

            # -------------------------------------------------------------------
            #prev_JAs = np.zeros(4)
            #dt = self.env.dt
            #desiredJointAngles = np.array([-2*np.pi/3, np.pi/4, -np.pi/4, np.pi])

            #detectedJointVels = self.angle_normalize(detectedJointAngles - prev_JAs)/dt
            #prev_JAs = detectedJointAngles
            #self.env.step((detectedJointAngles,detectedJointVels,desiredJointAngles,np.zeros(4)))

            #print(self.env.ground_truth_joint_velocities - detectedJointVels)

            # --------------------------------------------------------
            #velocity end

            true_JV = self.env.ground_truth_joint_velocities
            true_ee = self.env.ground_truth_end_effector
            true_JA = self.env.ground_truth_joint_angles

            prev_jvs.append(self.angle_normalize(detectedJointAngles-prev_JAs))
            detectedJointVels = (sum(prev_jvs)/len(prev_jvs))/dt
            prev_JAs = detectedJointAngles

            ee_pos = self.detect_ee(arrxy_HSV,arrxz_HSV,Vpeak)
            ee_vel = (ee_pos - np.squeeze(np.array(prevEePos)))/dt
            prevEePos = ee_pos

            ee_target = self.detect_target(arrxy_HSV,arrxz_HSV,Vpeak)

            switch = switch or (np.linalg.norm(ee_target - ee_target_pre) >= 0.2)

            timeused += dt

            if((np.linalg.norm(ee_target - ee_target_pre) >= 0.2)):
                print timeused



            ee_target_pre = ee_target

            #desiredJointAngles = true_JA+self.IK(detectedJointAngles,ee_target)

            #print(switch)
            #print(desiredJointAngles)

            J = self.Jacobian(detectedJointAngles)[0:3,:]

            ee_desired_force = self.ts_pd_control(ee_pos, ee_vel, ee_target)
            torques = J.T*ee_desired_force #+ grav_opposite_torques
            #ee_desired_force = self.js_pd_control(detectedJointAngles, detectedJointVels, desiredJointAngles)

            #torques = ee_desired_force + grav_opposite_torques

            # test
            #self.detect_target(arrxy_HSV,arrxz_HSV,Vpeak)

            #J = self.Jacobian(true_JA)
            #edot = np.dot(J[0:3],detectedJointVels)
            #print(true_ee)


            # Etest
            if(np.linalg.norm(ee_pos-ee_target) <= 0.15):
                ee_desired_force = self.js_pd_control(detectedJointAngles, detectedJointVels, detectedJointAngles)
                self.env.step((np.zeros(3),np.zeros(3),np.zeros(3),ee_desired_force))
            else:
                ee_desired_force = self.ts_pd_control(ee_pos, ee_vel, ee_target)
                torques = J.T*ee_desired_force #+ grav_opposite_torques
                self.env.step((np.zeros(3),np.zeros(3),np.zeros(3),torques))


            #if(np.dot(self.detect_red(arrxy_HSV,arrxz_HSV),ee_target) < 0):
            #    self.env.step((np.zeros(3),np.zeros(3),np.zeros(3),[torques[0]+50,torques[1],torques[2],torques[3]]))
            #else:
            #    self.env.step((np.zeros(3),np.zeros(3),np.zeros(3),torques))


            #self.env.step((np.zeros(4),np.zeros(4),np.zeros(4), np.zeros(4)))
            #The step method will send the control input to the robot, the parameters are as follows: (Current Joint Angles/Error, Current Joint Velocities, Desired Joint Angles, Torque input)

#main method
def main():
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()
