
# coding: utf-8

# In[1]:


import numpy as np
def get_homograph(u,v):
    A = np.array([[u[0][0], u[0][1], 1, 0, 0, 0, -1 * u[0][0] * v[0][0], -1 * u[0][1] * v[0][0]],
                  [0, 0, 0, u[0][0], u[0][1], 1, -1 * u[0][0] * v[0][1], -1 * u[0][1] * v[0][1]],
                  [u[1][0], u[1][1], 1, 0, 0, 0, -1 * u[1][0] * v[1][0], -1 * u[1][1] * v[1][0]],
                  [0, 0, 0, u[1][0], u[1][1], 1, -1 * u[1][0] * v[1][1], -1 * u[1][1] * v[1][1]],
                  [u[2][0], u[2][1], 1, 0, 0, 0, -1 * u[2][0] * v[2][0], -1 * u[2][1] * v[2][0]],
                  [0, 0, 0, u[2][0], u[2][1], 1, -1 * u[2][0] * v[2][1], -1 * u[2][1] * v[2][1]],
                  [u[3][0], u[3][1], 1, 0, 0, 0, -1 * u[3][0] * v[3][0], -1 * u[3][1] * v[3][0]],
                  [0, 0, 0, u[3][0], u[3][1], 1, -1 * u[3][0] * v[3][1], -1 * u[3][1] * v[3][1]]
                ])
    b = np.array([[v[0][0]],
                  [v[0][1]],
                  [v[1][0]],
                  [v[1][1]],
                  [v[2][0]],
                  [v[2][1]],
                  [v[3][0]],
                  [v[3][1]]
                ])
    tmp = np.dot(np.linalg.inv(A), b)
    H = np.array([[tmp[0][0], tmp[1][0], tmp[2][0]],
                  [tmp[3][0], tmp[4][0], tmp[5][0]],
                  [tmp[6][0], tmp[7][0], 1]
                 ])
    return H


def interpolation(img, new_x, new_y):
    fx = round(new_x - int(new_x), 2)
    fy = round(new_y - int(new_y), 2)
    p = np.zeros((3,))
    p += (1 - fx) * (1 - fy) * img[int(new_y), int(new_x)]
    p += (1 - fx) * fy * img[int(new_y) + 1, int(new_x)]
    p += fx * (1 - fy) * img[int(new_y), int(new_x) + 1]
    p += fx * fy * img[int(new_y) + 1, int(new_x) + 1]
    return p

def forward_warping(u,v,input_image,canvas):
    matrix = get_homograph(u,v)
    i0_max = u[0:4,0:1].max()
    i0_min = u[0:4,0:1].min()
    i1_max = u[0:4,1:2].max()
    i1_min = u[0:4,1:2].min()
    i0_range = i0_max-i0_min
    i1_range = i1_max-i1_min
    
    for i in range(i1_range):
        for j in range(i0_range):
            tmp2 = np.dot(matrix, np.array([[j+i0_min, i+i1_min, 1]]).T)
            x, y = int(tmp2[0][0] / tmp2[2][0]), int(tmp2[1][0] / tmp2[2][0])
            canvas[y][x] = input_image[i+i1_min][j+i0_min]
    return canvas

def backward_warping(u,v,input_image,canvas):
    matrix = get_homograph(u,v) # v: output, u: input
    i0_max = u[0:4,0:1].max()
    i0_min = u[0:4,0:1].min()
    i1_max = u[0:4,1:2].max()
    i1_min = u[0:4,1:2].min()
    i0_range = i0_max-i0_min
    i1_range = i1_max-i1_min
    for j in range(i1_range):
        for i in range(i0_range):
            new_pos = np.dot(matrix, np.array([[i+i0_min, j+i1_min, 1]]).T)
            new_x, new_y = new_pos[0][0] / new_pos[2][0], new_pos[1][0] / new_pos[2][0]
            res = interpolation(input_image, new_x, new_y)
            canvas[j+i1_min][i+i0_min] = res
    return canvas