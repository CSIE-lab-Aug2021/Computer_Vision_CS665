
## HW2-1 Camera Calibration 程式說明
### 1. 主檔案: `hw2-1.py`
### 2. 本次作業Function 寫在主檔案內, 內容包含:
> #### `Projection_Matrix`: 產生 Projection Matrix
> #### `KRt`: 根據Projection Matrix, 產生intrinsic matrices K, rotation matrices R and translation vectors t
> #### `ReProject2D`: 根據K, R, t, 重新投影2D點, 算出root-mean-squared errors
> #### `Project`: 根據2D投影點, 在原始圖上畫出點

### 3. 檔案內附上助教提供的 `visualize.py`, 用來畫出照相機位置
### 4. 有使用的Library: `cv2`: 讀取/存取圖片, `numpy`: 矩陣運算,  `scipy`: rq分解, `matplotlib`: 畫圖
### 5. 直接執行`python hw2-1.py`, 就會在folder " `output` " 裡面輸出資料如下:
> #### (1) 6張圖片: ReProject2D of chessboard_1, ReProject2D of chessboard_2 (題目給的圖片), ReProject2D of image1, ReProject2D of image2 (自己拍攝的圖片), corner detection of image1 & image2 (利用corner detection找的點, 所畫出來的圖片)
> #### (2) 10個文字檔(題目給的圖片, 所算出來的相關矩陣): Projection Matrix of chessboard_1, Projection Matrix of chessboard_2, intrinsic matrices of chessboard_1, intrinsic matrices of chessboard_2, rotation matrices of chessboard_1, rotation matrices of chessboard_2, translation vectors of chessboard_1, translation vectors of chessboard_2, RMSE of chessboard_1, RMSE of chessboard_2, 
> #### (3) 4個文字檔(自己拍的圖片, 所算出來的相關矩陣): Projection Matrix of image1, Projection Matrix of image2, RMSE of image1, RMSE of image2
