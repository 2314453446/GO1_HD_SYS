import cv2

# 读取彩色图像
image = cv2.imread('img/00101.jpg')

# 将彩色图像转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 保存灰度图像
cv2.imwrite('img/gray_image.jpg', gray_image)

# 显示图像
cv2.imshow('Gray Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
