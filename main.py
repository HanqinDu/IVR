#!/usr/bin/env python2.7
import gym
import reacher3D.Reacher
import numpy as np
import cv2
import math
import scipy as sp
import collections
import time
import classifier



class MainReacher():
#    def __init__(self):
#        self.env = gym.make('3DReacherMy-v0')
#        self.env.reset()
#for target classifier [eris]
    def __init__(self):
        #original
        self.env = gym.make('3DReacherMy-v0')
        self.env.reset()

    ##By eris
    def image_normalize(self,image):
        image = cv2.normalize(image,None,0,255,cv2.NORM_MINMAX)
        return image

    #By Du
    def coordinate_convert_3D(self,pixels):
        #Converts pixels into metres
        return np.array([(pixels[0]-self.env.viewerSize/2)/self.env.resolution,-(pixels[1]-self.env.viewerSize/2)/self.env.resolution,-(pixels[2]-self.env.viewerSize/2)/self.env.resolution])

    def coordinate_convert(self,pixels):
        #Converts pixels into metres
        return np.array([(pixels[0]-self.env.viewerSize/2)/self.env.resolution,-(pixels[1]-self.env.viewerSize/2)/self.env.resolution])

    def angle_(self, v1, v2):
        output = (np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        return output

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
        Jac = np.matrix(self.Jacobian_analytic(current_joint_angles))[0:3,:]
        Jac_inv = Jac.T
        #If the Jacobian is low rank (<2) then use the transpose otherwise use psuedo-inverse
        if(np.linalg.matrix_rank(Jac,0.4)<2):
            Jac_inv = Jac.T
        else:
            Jac_inv = Jac.T*np.linalg.inv(Jac*Jac.T)
        #Apply inverted jacobian on the position error in task space to get qdot
        q_dot = Jac_inv*np.matrix(pos_error).T
        return np.squeeze(np.array(q_dot.T))

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
        y_vector = np.array([0,1,0])
        #Calculate geometric jacobian using equations
        pos_3D = np.zeros(3)

        pos_3D[0:3] = (ee_pos-j1_pos).T
        jacobian[0:3,0] = np.cross(y_vector,pos_3D)[0:3]
        pos_3D[0:3] = (ee_pos-j2_pos).T
        jacobian[0:3,1] = np.cross(z_vector,pos_3D)[0:3]
        pos_3D[0:3] = (ee_pos-j3_pos).T
        jacobian[0:3,2] = np.cross(z_vector,pos_3D)[0:3]
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

#By Du


    #By eris
    def detect_joint_angles(self,image_xy,image_xz,Vpeak):
        #Calculate the relevant joint angles from the image
        #Obtain the center of each coloured blob(red green blue dblue)
        jointPos1 = self.detect_red(image_xy,image_xz)
        jointPos2 = self.detect_green(image_xy,image_xz)
        jointPos3 = self.detect_blue(image_xy,image_xz,Vpeak)
        jointPos4 = self.detect_ee(image_xy,image_xz,Vpeak)

        #print(jointPos4)
        #Solve using trigonometry
        ja1 = self.angle_(jointPos1,np.array([1,0,0]))
        ja2 = self.angle_(jointPos2-jointPos1,jointPos1)
        ja3 = self.angle_(jointPos3-jointPos2,jointPos2-jointPos1)
        ja4 = self.angle_(jointPos4-jointPos3,jointPos3-jointPos2)

        ja1 = self.angle_normalize(ja1)
        ja2 = self.angle_normalize(ja2)
        ja3 = self.angle_normalize(ja3)
        ja4 = self.angle_normalize(ja4)

        return np.array([ja1,ja2,ja3,ja4])

    def angle_normalize(self,x):
        #Normalizes the angle between pi and -pi
        return (((x+np.pi) % (2*np.pi)) - np.pi)
#detect target by eris

    def detect_target(self,image_xy,image_xz,classifier):
        #Detect the center of the target circle (Colour: [200,200,200])
        #SAME AS DETECT_BLUE JUST WITH DIFFERENT COLOUR LIMITS

        kernel = np.ones((5,5),np.uint8)
        #image_xy
        mask_xy = cv2.inRange(image_xy,(170,170,170),(189,189,189))#178.5 target
        mask_xy = cv2.dilate(mask_xy,kernel,iterations=3)
        M_xy = cv2.moments(mask_xy)
        #use findCountours to find center
        imgXY,countour_XY, _ = cv2.findContours(mask_xy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cxy1 = np.reshape(np.mean(countour_XY[0],0,dtype = np.int),2)
        cxy2 = np.reshape(np.mean(countour_XY[1],0,dtype = np.int),2)

        #image_xz
        mask_xz = cv2.inRange(image_xz,(170,170,170),(189,189,189))#178.5 target
        mask_xz = cv2.dilate(mask_xz,kernel,iterations=3)
        M_xz = cv2.moments(mask_xz)
        #use findCountours to find center
        imgXZ,countour_XZ, _ = cv2.findContours(mask_xz, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cxz1 = np.reshape(np.mean(countour_XZ[0],0,dtype = np.int),2)
        cxz2 = np.reshape(np.mean(countour_XZ[1],0,dtype = np.int),2)

        #to form each coordinate
        XY1 = self.coordinate_convert(cxy1)
        XY2 = self.coordinate_convert(cxy2)
        XZ1 = self.coordinate_convert(cxz1)
        XZ2 = self.coordinate_convert(cxz2)

        target1 = np.array([XY1[0],XY1[1],XZ1[1]])
        target2 = np.array([XY2[0],XY2[1],XZ2[1]])

        right=cxy1[0]+20
    	if right>np.shape(mask_xy)[1]:
    		right=np.shape(mask_xy)[1]
    	left=cxy1[0]-20
    	if left<0:
    		left=0
    	down=cxy1[1]+20
    	if down>np.shape(mask_xy)[0]:
    		down=np.shape(mask_xy)[0]
    	up=cxy1[1]-20
    	if up<0:
    		up=0

        test_img=mask_xy[up:down, left:right]
    	cv2.imwrite('test.jpg', test_img)
    	cv2.waitKey(0)


        _, pred = classifier.classify('test.jpg')
        pred = pred[0][0]

        if (pred ==0):
            return (target1,target2)
        else:
            return (target2,target1)
    #By eris end

    #Gravity by eris
    def grav(self, current_joint_angles):##gai
        # Gravitational acceleration and mass
        g = 9.81
        m = 1
        # Opposite torque from gravity in link 1
        t1 = 5*m*g/2*math.cos(current_joint_angles[0]) + 3*m*g/2*math.cos(current_joint_angles[0] + current_joint_angles[1]) + m*g/2*math.cos(current_joint_angles[0] + current_joint_angles[1] + current_joint_angles[2])
        # Opposite torque from gravity in link 2
        t2 = 3*m*g/2*math.cos(current_joint_angles[0] + current_joint_angles[1]) + m*g/2*math.cos(current_joint_angles[0] + current_joint_angles[1] + current_joint_angles[2])
        # Opposite torque from gravity in link 3
        t3 = m*g/2*math.cos(current_joint_angles[0] + current_joint_angles[1] + current_joint_angles[2])
        return np.matrix([t1, t2, t3]).T
    #gravity eris

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


        #detect target eris
        test_classifier = classifier.main()

        #end target


        for loop in range(100000):

            #self.env.render returns 2 RGB arrays of the robot, one for the xy-plane, and one for the xz-plane
            arrxy,arrxz = self.env.render('rgb-array')


            # D: calculate joint position
            # Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].

            arrxy_HSV = cv2.cvtColor(arrxy, cv2.COLOR_BGR2HSV)
            arrxz_HSV = cv2.cvtColor(arrxz, cv2.COLOR_BGR2HSV)

            norm_image_xy = self.image_normalize(arrxy)#####
            norm_image_xz = self.image_normalize(arrxz)
            #target
            self.detect_target(norm_image_xy,norm_image_xz,test_classifier)

            img_hist=cv2.calcHist([arrxy],[2],None,[256],[0,256])
            Vpeak = np.argmax(img_hist)

            detectedJointAngles = self.detect_joint_angles(arrxy_HSV,arrxz_HSV,Vpeak)
            #print(self.env.ground_truth_joint_angles - detectedJointAngles)

            #velocity part [eris]
            #The change in time between iterations can be found in the self.env.dt variable

            # -------------------------------------------------------------------
            prev_JAs = np.zeros(4)
            dt = self.env.dt
            desiredJointAngles = np.array([-2*np.pi/3, np.pi/4, -np.pi/4, np.pi])

            detectedJointVels = self.angle_normalize(detectedJointAngles - prev_JAs)/dt
            prev_JAs = detectedJointAngles

            self.env.step((detectedJointAngles,detectedJointVels,desiredJointAngles,np.zeros(4)))



            # --------------------------------------------------------
            #velocity end


            #jointAngles = np.array([0.5,0.5,0.5,-0.5])#####du

            jointAngles = np.array([1.5,1.5,1.5,1.5])#####du

            self.env.step((np.zeros(4),np.zeros(4),jointAngles, np.zeros(4)))######du

            #self.env.step((np.zeros(4),np.zeros(4),np.zeros(4), np.zeros(4)))
            #The step method will send the control input to the robot, the parameters are as follows: (Current Joint Angles/Error, Current Joint Velocities, Desired Joint Angles, Torque input)

#main method
def main():
    reach = MainReacher()
    reach.go()




if __name__ == "__main__":
    main()
