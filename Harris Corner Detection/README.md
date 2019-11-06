
# Image Formation and Features
### CS655000 Computer Vision Homework 1
### Brief
* Due: Wed, 10/16, 23:59
* Use Python to complete the homework.
* If you encounter any problem, let’s discuss on iLMS instead of email.

## Part 1. Harris Corner Detection

With the Harris corner detector described in slides (p.79), mark the detected corners on the image.

<img style="float: left;" src="Harris Corner Detection/1.PNG" width="60%">


### A. Functions:
* `gaussian_smooth()`: filter images with Gaussian blur.
* `sobel_edge_detection()`: apply the Sobel filters to the blurred images and compute the magnitude and direction of gradient. (You should eliminate weak gradients by proper threshold.)
* `structure_tensor()`: use the gradient magnitude above to compute the structure tensor (second-moment matrix).
* `nms()`: perform non-maximal suppression on the results above along with appropriate threshold for corner detection.

### B. Results:
* a. Original image
    * i. Gaussian smooth results: 𝜎=5 and kernel size=5 and 10 (**2 images**)
    * ii. Sobel edge detection results
     * (1) magnitude of gradient (Gaussian kernel size=5 and 10) (**2 images**)
     * (2) direction of gradient (Gaussian kernel size=5 and 10) (**2 images**)
        (You can choose arbitrary color map to display)
    * iii. Structure tensor + NMS results (Gaussian kernel size=10)
     * (1) window size = 3x3 (**1 image**)
     * (2) window size = 30x30 (**1 image**)
* b. Final results of rotating (by 30°) original images (**1 image**)
* c. Final results of scaling (to 0.5x) original images (**1 image**)

### C. Report:
* a. Discuss the results of blurred images and detected edges between different kernel sizes of Gaussian filter.
* b. Discuss the difference between 3x3 and 30x30 window sizes of structure tensor.
* c. Discuss the effect of non-maximal suppression.
* d. Discuss the results of rotated and scaled image. Is Harris detector rotation-invariant or scale-invariant? Explain the reason.

### D. Notice:
* a. You should **NOT** use any functions which can get the result directly in each steps. (`cv2.Sobel`, `cv2.Laplacian`, `cv2.cornerHarris`, `skimg.feature.local_binary_pattern`, etc.)
* b. Your code should display and output image results mentioned above.
* c. You should provide a README file about your execution instructions.
