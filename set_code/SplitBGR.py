import cv2
import numpy as np

# 读取图像
image = cv2.imread('../Dataset/4/56.jpg')

# 分离通道
blue_channel, green_channel, red_channel = cv2.split(image)

# 设置阈值
thresh_value_blue = 50  # 蓝色通道阈值
thresh_value_green = 50  # 绿色通道阈值
thresh_value_red = 6  # 红色通道阈值

# 对蓝色通道进行阈值处理
_, blue_thresh = cv2.threshold(blue_channel, thresh_value_blue, 255, cv2.THRESH_BINARY)

# 对绿色通道进行阈值处理
_, green_thresh = cv2.threshold(green_channel, thresh_value_green, 255, cv2.THRESH_BINARY)

# 对红色通道进行阈值处理
_, red_thresh = cv2.threshold(red_channel, thresh_value_red, 255, cv2.THRESH_BINARY)

# 合并阈值处理后的通道
thresh_blue_image = cv2.merge([blue_thresh, np.zeros_like(green_channel), np.zeros_like(red_channel)])
thresh_green_image = cv2.merge([np.zeros_like(blue_channel), green_thresh, np.zeros_like(red_channel)])
thresh_red_image = cv2.merge([np.zeros_like(blue_channel), np.zeros_like(green_channel), red_thresh])

# 显示原始图像和阈值处理后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Blue Channel', thresh_blue_image)
cv2.imshow('Thresholded Green Channel', thresh_green_image)
cv2.imshow('Thresholded Red Channel', thresh_red_image)

# 等待用户按下任意键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
