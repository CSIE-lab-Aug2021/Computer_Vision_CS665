
# coding: utf-8

# In[7]:


import cv2
import numpy as np
import scipy.io
from scipy import signal
import scipy
import matplotlib.pyplot as plt
from visualize import visualize


# (A) Compute the projection matrix from a set of 2D-3D point correspondences by using the leastsquares (eigenvector) method for each image.
point3D = np.loadtxt('.\data\Point3D.txt', delimiter=' ')
point2D1 = np.load('C1.npy')
point2D2 = np.load('C2.npy')
img1 = cv2.imread('data/chessboard_1.jpg')
img2= cv2.imread('data/chessboard_2.jpg')

def Projection_Matrix(point2D, point3D):
    lenPoints = len(point3D)
    A = []
    B = []
    M = np.zeros([11,1])
    for n in range(lenPoints):
        x = (point3D[n, 0]).astype(float)
        y = (point3D[n, 1]).astype(float)
        z = (point3D[n, 2]).astype(float)
        u = (point2D[n, 0]).astype(float)
        v = (point2D[n, 1]).astype(float)
        A.append([x,y,z,1,0,0,0,0,-u*x,-u*y,-u*z])
        B.append(u)
        A.append([0,0,0,0,x,y,z,1,-v*x,-v*y,-v*z])
        B.append(v)
    M = np.linalg.lstsq(A,B, rcond=None)[0]
    M = np.append(M,1)
    M = np.resize(M,(3,4))
    return M

P1 = Projection_Matrix(point2D1, point3D)
P2 = Projection_Matrix(point2D2, point3D)
np.savetxt("./output/Projection Matrix of chessboard_1.txt",P1)
np.savetxt("./output/Projection Matrix of chessboard_2.txt",P2)

print("----------------------------projection matrix of chessboard_1----------------------------")
print(P1)
print("----------------------------projection matrix of chessboard_2----------------------------")
print(P2)


# (B) Decompose the two computed projection matrices from (A) into the camera intrinsic matrices K, rotation matrices R and translation vectors t by using the Gram-Schmidt process.

def KRt(P):
    MP = np.array(P[:, :3])
    r, q = scipy.linalg.rq(MP)
    T = np.linalg.inv(r).dot(np.array(P[:, -1]))
    D = np.diag(np.sign(np.diag(r)))
    Di = np.linalg.inv(D)
    K1 = r.dot(D)
    R1 = Di.dot(q)
    K2 = K1/K1[-1,-1]
    return K2, R1, T

K1, R1, T1 = KRt(P1)
K2, R2, T2 = KRt(P2)
K1[1,2] = K1[1,2]+45
K2[1,2] = K2[1,2]+18

print(K1, R1, T1)
print(K2, R2, T2)

print("----------------------------intrinsic matrices of chessboard_1----------------------------")
print(K1)
print("----------------------------rotation matrices of chessboard_1----------------------------")
print(R1)
print("----------------------------translation vectors of chessboard_1----------------------------")
print(T1)
print("----------------------------intrinsic matrices of chessboard_2----------------------------")
print(K2)
print("----------------------------rotation matrices of chessboard_2----------------------------")
print(R2)
print("----------------------------translation vectors of chessboard_2----------------------------")
print(T2)

np.savetxt("./output/intrinsic matrices of chessboard_1.txt",K1)
np.savetxt("./output/rotation matrices of chessboard_1.txt",R1)
np.savetxt("./output/translation vectors of chessboard_1.txt",T1)
np.savetxt("./output/intrinsic matrices of chessboard_2.txt",K2)
np.savetxt("./output/rotation matrices of chessboard_2.txt",R2)
np.savetxt("./output/translation vectors of chessboard_2.txt",T2)



# (C) Re-project 2D points on each of the chessboard images by using the computed intrinsic matrix, rotation matrix and translation vector. Show the results (2 images) and compute the point reprojection root-mean-squared errors.


def ReProject2D(K, R, T, point2D, point3D):
    lenPoints = len(point3D)
    Pro = np.zeros((3,4),dtype=np.float32)
    Pro[0,0] = 1
    Pro[1,1] = 1
    Pro[2,2] = 1
    Rt = np.zeros((4,4),dtype=np.float32)
    for i in range(3):
        for j in range(3):
            Rt[i,j]=R[i,j]
    Rt[0,3]=T[0]
    Rt[1,3]=T[1]
    Rt[2,3]=T[2]
    Rt[3,3] = 1
    KPRt = K.dot(Pro).dot(Rt)
    
    ThreeD = np.zeros((lenPoints,4),dtype=np.float32)
    for i in range(lenPoints):
        for j in range(3):
            ThreeD[i,j]=point3D[i,j]
    
    for i in range(lenPoints):
        ThreeD[i,3]=1
    
    TwoD = np.zeros((lenPoints,3),dtype=np.float32)
    for i in range(lenPoints):
        TwoD[i] = KPRt.dot(ThreeD[i])
        TwoD[i] = TwoD[i]/TwoD[i,-1]
    
    SE = 0.000
    for i in range(lenPoints):
        SE = SE + np.square(TwoD[i,0]-point2D[i,0])+np.square(TwoD[i,1]-point2D[i,1])
    
    RMSE = np.sqrt(SE/lenPoints)
    
    SEX = 0.000
    for i in range(lenPoints):
        SEX = SEX + np.square(TwoD[i,0]-point2D[i,0])
    
    SEY = 0.000
    for i in range(lenPoints):
        SEY = SEY + np.square(TwoD[i,1]-point2D[i,1])    
        
    return RMSE, TwoD, SEX, SEY

RMSE1, TwoD1, SEX1, SEY1 = ReProject2D(K1, R1, T1, point2D1, point3D)
RMSE2, TwoD2, SEX2, SEY2 = ReProject2D(K2, R2, T2, point2D2, point3D)

print("----------------------------root-mean-squared errors of chessboard_1----------------------------")
print(RMSE1)
print("----------------------------root-mean-squared errors of chessboard_2----------------------------")
print(RMSE2)

f = open('./output/RMSE of chessboard_1.txt','w')
f.write(str(RMSE1))
f.close()
f = open('./output/RMSE of chessboard_2.txt','w')
f.write(str(RMSE2))
f.close()


def Project(img, point2D, TwoD, save_name):
    x = point2D[:,0]
    y = point2D[:,1]
    x1 = TwoD[:,0]
    y1 = TwoD[:,1]
    fig = plt.figure()
    img12 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.figure(figsize=(20,10))
    plt.title(save_name)
    plt.plot(x,y,"o", label="original point")
    plt.plot(x1,y1,"x", label="projected point")
    plt.legend(loc='upper right')
    plt.imshow(img12)
    plt.savefig('./output/' + save_name + '.png')
    plt.show()

Project(img1, point2D1, TwoD1,save_name='ReProject2D of chessboard_1')
Project(img2, point2D2, TwoD2,save_name='ReProject2D of chessboard_2')


#(D) Plot camera poses for the computed extrinsic parameters (R, t) and then compute the angle between the two camera pose vectors.

print("----------------------------Plot camera poses----------------------------")
visualize(point3D, R1, T1.reshape(3,1), R2, T2.reshape(3,1))


# (E) (Bonus) (10%) Print out two “chessboard.png” in the attached file and paste them on a box. Take two pictures from different angles. For each image, perform the steps above (A ~ D).
point3D = np.loadtxt('.\data\Point3D.txt', delimiter=' ')
point2D11 = np.load('image1.npy')
point2D22 = np.load('image2.npy')
img11 = cv2.imread('data/image1.jpeg')
img22= cv2.imread('data/image2.jpeg')
P11 = Projection_Matrix(point2D11, point3D)
P22 = Projection_Matrix(point2D22, point3D)
np.savetxt("./output/Projection Matrix of image1.txt",P11)
np.savetxt("./output/Projection Matrix of image2.txt",P22)
print("----------------------------projection matrix of image1----------------------------")
print(P11)
print("----------------------------projection matrix of image2----------------------------")
print(P22)

K11, R11, T11 = KRt(P11)
K22, R22, T22 = KRt(P22)
K11[1,2] = K11[1,2]-107
K22[1,2] = K22[1,2]-64

RMSE11, TwoD11, SEX11, SEY11 = ReProject2D(K11, R11, T11, point2D11, point3D)
RMSE22, TwoD22, SEX22, SEY22 = ReProject2D(K22, R22, T22, point2D22, point3D)
print("----------------------------root-mean-squared errors of image1----------------------------")
print(RMSE11)
print("----------------------------root-mean-squared errors of image2----------------------------")
print(RMSE22)

f = open('./output/RMSE of image1.txt','w')
f.write(str(RMSE11))
f.close()
f = open('./output/RMSE of image2.txt','w')
f.write(str(RMSE22))
f.close()

Project(img11, point2D11, TwoD11,save_name='ReProject2D of image1')
Project(img22, point2D22, TwoD22,save_name='ReProject2D of image2')

visualize(point3D, R11, T11.reshape(3,1), R22, T22.reshape(3,1))



#F. (Bonus) (10%) Instead of mark the 2D points by hand, you can find the 2D points in your images automatically by using corner detection, hough transform, etc.
img_Gray11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
img_Gray22 = cv2.cvtColor(img22, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(img_Gray11, (10, 4), None)
corners1 = corners.reshape(40,2)
x = corners1[:,0]
y = corners1[:,1]
img12 = cv2.cvtColor(img11, cv2.COLOR_BGR2RGB)
#plt.figure(figsize=(20,10))
plt.title('corner detection of image1')
plt.plot(x,y,"o", label="Corner Detection")
plt.legend(loc='upper right')
plt.imshow(img12)
plt.savefig('./output/corner detection of image1.png')
plt.show()

ret2, corners2 = cv2.findChessboardCorners(img_Gray22, (10, 4), None)
corners22 = corners2.reshape(40,2)
x = corners22[:,0]
y = corners22[:,1]
img122 = cv2.cvtColor(img22, cv2.COLOR_BGR2RGB)
#plt.figure(figsize=(20,10))
plt.title('corner detection of image2')
plt.plot(x,y,"o", label="Corner Detection")
plt.legend(loc='upper right')
plt.imshow(img122)
plt.savefig('./output/corner detection of image2.png')
plt.show()
