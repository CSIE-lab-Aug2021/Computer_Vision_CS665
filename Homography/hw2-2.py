
# coding: utf-8

# In[3]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from homography import forward_warping, backward_warping

# Image-A Forward Warping
screen_corner1 = np.load('11.npy')
screen_corner2 = np.load('12.npy')

## left to right
canvas = cv2.imread('./input/imga.jpg')
input_image = cv2.imread('./input/imga.jpg')
v = screen_corner2
u = screen_corner1
canvas = forward_warping(u,v,input_image,canvas)
## right to left
v = screen_corner1
u = screen_corner2
input_image = cv2.imread('./input/imgA.jpg')
canvas = forward_warping(u,v,input_image,canvas)

save_img_path = os.path.join('./output', 'forward_warping_imgA.png')
cv2.imwrite(save_img_path, canvas)

# Image-A Backward Warping
canvas = cv2.imread('./input/imga.jpg')
input_image = cv2.imread('./input/imga.jpg')
## left to right
v = screen_corner1
u = screen_corner2
canvas = backward_warping(u,v,input_image,canvas)
## right to left
v = screen_corner2
u = screen_corner1
canvas = backward_warping(u,v,input_image,canvas)

save_img_path = os.path.join('./output', 'backward_warping_imgA.png')
cv2.imwrite(save_img_path, canvas)

# Image-B/C Forward Warping
canvas1 = cv2.imread('./input/imgb.jpeg')
canvas2 = cv2.imread('./input/imgc.jpeg')
screen_corner1 = np.load('b.npy')
screen_corner2 = np.load('c.npy')

## Forward Warping-imgb
v = screen_corner2
u = screen_corner1
input_image = cv2.imread('./input/imgb.jpeg')
canvas = forward_warping(u,v,input_image,canvas2)
save_img_path = os.path.join('./output', 'forward_warping_imgb.png')
cv2.imwrite(save_img_path, canvas)

## Forward Warping-imgc
v = screen_corner1
u = screen_corner2
input_image = cv2.imread('./input/imgc.jpeg')
canvas = forward_warping(u,v,input_image,canvas1)
save_img_path = os.path.join('./output', 'forward_warping_imgc.png')
cv2.imwrite(save_img_path, canvas)

# Image-B/C Backward Warping
## Backward Warping-imgb
canvas = cv2.imread('./input/imgc.jpeg')
input_image = cv2.imread('./input/imgb.jpeg')
v = screen_corner1
u = screen_corner2
canvas2 = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
canvas = backward_warping(u,v,input_image,canvas)
save_img_path = os.path.join('./output', 'backward_warping_imgb.png')
cv2.imwrite(save_img_path, canvas)

## Backward Warping-imgc
canvas = cv2.imread('./input/imgb.jpeg')
input_image = cv2.imread('./input/imgc.jpeg')
v = screen_corner2
u = screen_corner1
canvas = backward_warping(u,v,input_image,canvas)
save_img_path = os.path.join('./output', 'backward_warping_imgc.png')
cv2.imwrite(save_img_path, canvas)

