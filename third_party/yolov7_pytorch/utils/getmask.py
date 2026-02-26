import cv2
import numpy as np
import matplotlib.pyplot as plt


def weed_detection(image_path):
    # Step 1: Read the input image
    image = cv2.imread(image_path)

    # Step 2: Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 3: Create a mask based on the green color range in the HSV color space
    lower_green = np.array([25, 40, 40])  # Adjust the lower and upper thresholds as needed
    upper_green = np.array([75, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Step 4: Find contours in the mask
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Create an output image with black background and weed potential areas in white
    output_image = np.zeros_like(image)
    output_image[mask != 0] = (255, 255, 255)  # Set the weed potential areas to white

    # Step 6: Draw rectangles around the detected contours
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangles
    #
    # # Show the original image with rectangles around the weed potential areas
    # cv2.imshow("Detected Weed Areas", output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return output_image


# 形态学操作类
class morphological_trans:
    _defaults = {
        "cross_kernel": np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]], dtype=np.uint8)
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
        pass

    # 膨胀操作
    def dilate(self, image, kernel_size=(3, 3), iterations=1, special_kernel="kernel"):
        kernel = np.ones(kernel_size, np.uint8)
        if special_kernel != "kernel":
            kernel = getattr(self, special_kernel)
        dilated_image = cv2.dilate(image, kernel=kernel, iterations=iterations)
        return dilated_image

    def erode(self, image, kernel_size=(3, 3), iterations=1, special_kernel="kernel"):
        kernel = np.ones(kernel_size, np.uint8)
        if special_kernel != "kernel":
            kernel = getattr(self, special_kernel)
        eroded_image = cv2.erode(image, kernel, iterations=iterations)
        return eroded_image

    def open(self, image, kernel_size=(3, 3), special_kernel="kernel"):
        kernel = np.ones(kernel_size, np.uint8)
        if special_kernel != "kernel":
            kernel = getattr(self, special_kernel)
        opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return opened_image

    def close(self, image, kernel_size=(3, 3), special_kernel="kernel", iterations=1):
        kernel = np.ones(kernel_size, np.uint8)
        if special_kernel != "kernel":
            kernel = getattr(self, special_kernel)
        closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        return closed_image

    def gradient(self, image, kernel_size=(3, 3)):
        kernel = np.ones(kernel_size, np.uint8)
        gradient_image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        return gradient_image

    def top_hat(self, image, kernel_size=(3, 3)):
        kernel = np.ones(kernel_size, np.uint8)
        top_hat_image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        return top_hat_image

    def black_hat(self, image, kernel_size=(3, 3)):
        kernel = np.ones(kernel_size, np.uint8)
        black_hat_image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        return black_hat_image


def imshow(showimage, title='title'):
    cv2.imshow(title, showimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def remove_small_areas(mask, min_area):
    """
    从二进制掩码图像中移除小于指定面积的区域。

    :param mask: 输入的二进制掩码图像
    :param min_area: 最小保留区域的面积阈值
    :return: 移除小区域后的掩码图像
    """
    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空白掩码图像，与输入图像具有相同的大小和深度
    result_mask = np.zeros_like(mask)

    # 循环遍历每个轮廓
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)

        # 如果面积大于等于最小面积阈值，则保留该轮廓
        if area >= min_area:
            cv2.drawContours(result_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return result_mask


# def get_box(img):
#     edged = cv2.Canny(img,threshold1=30,threshold2=100)
#     contours,_ =cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     # 循环遍历所有轮廓
#     for contour in contours:
#         # 获取边界框坐标
#         x, y, w, h = cv2.boundingRect(contour)
#
#         # 在原始图像上绘制边界框
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

def get_contour_gravity_point(contour,image):
    # Calculate the moment of contour
    M = cv2.moments(contour)

    # Calculate the centroid of the contour
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        print(f'Centroid of the contour is at: ({cx}, {cy})')

        # Draw centroid on the image (optional)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
    return image

# 在原图上画出 segmentation的透明图层
def draw_overlayer(contour,image,color=(255,255,255)):
    # 创建一个透明层，大小和原始图像相同
    overlay = np.zeros_like(image, dtype=np.uint8)

    # 在透明层上绘制轮廓（使用白色）
    cv2.drawContours(overlay, [contour], color, -1)  # idx是要绘制的轮廓的索引

    # 定义透明度（alpha），在 0 和 1 之间
    alpha = 0.5

    # 将透明层叠加到原始图像上
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

# 用于检测单个目标的重心
def get_single_predict(img_path):
    image_path = img_path
    morph = morphological_trans()
    hsvimg = weed_detection(image_path)
    hsvimg = hsvimg[:, :, 0]
    return hsvimg

def get_mask(img_path):
    image_path = img_path
    morph = morphological_trans()
    hsvimg = weed_detection(image_path)
    hsvimg = hsvimg[:, :, 0]
    small_mask_removed = remove_small_areas(hsvimg, min_area=10)
    dilated_img = morph.dilate(small_mask_removed, iterations=3, kernel_size=(3, 3), special_kernel="kernel")
    eroded_img = morph.erode(dilated_img, kernel_size=(3, 3), special_kernel="kernel", iterations=3)
    # imshow(eroded_img)
    return eroded_img

def get_mask_yolov7(croped_image):
    # Step 2: Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(croped_image, cv2.COLOR_RGB2HSV)

    # Step 3: Create a mask based on the green color range in the HSV color space
    lower_green = np.array([25, 40, 40])  # Adjust the lower and upper thresholds as needed
    upper_green = np.array([75, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Step 4: Find contours in the mask
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Create an output image with black background and weed potential areas in white
    output_image = np.zeros_like(croped_image)
    output_image[mask != 0] = (255, 255, 255)  # Set the weed potential areas to white


    return output_image



if __name__ == "__main__":
    image_path = r"D:\masterPROJECT\laser_weeding\weed_dataset\PhenoBench\val\images\06-05_00139_P0038051.png"  # Replace with the actual path to your image
    # ---------------------------
    mask = get_mask(image_path)
    imshow(mask)

    mask = get_mask(image_path)
    imshow(mask)
    # ----------------------------

    # # 出图
    # image = cv2.imread(image_path)
    # # Step 2: Convert the image to the HSV color space
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # # Step 3: Create a mask based on the green color range in the HSV color space
    # lower_green = np.array([18, 18, 18])  # Adjust the lower and upper thresholds as needed
    # upper_green = np.array([255, 255, 255])
    # mask = cv2.inRange(hsv_image, lower_green, upper_green)
    # # Step 4: Find contours in the mask
    # # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # Step 5: Create an output image with black background and weed potential areas in white
    # output_image = np.zeros_like(image)
    # output_image[mask != 0] = (255, 255, 255)
    #
    # morph = morphological_trans()
    # hsvimg = weed_detection(image_path)
    # hsvimg = hsvimg[:, :, 0]
    # small_mask_removed = remove_small_areas(hsvimg, min_area=10)
    # dilated_img = morph.dilate(small_mask_removed, iterations=8, kernel_size=(3, 3), special_kernel="kernel")
    # eroded_img = morph.erode(dilated_img, kernel_size=(3, 3), special_kernel="cross_kernel", iterations=8)
    # # imshow(eroded_img)
    # fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    # axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # axs[0].axis('off')
    # # axs[0].set_title("Original Image")
    #
    # axs[1].imshow(output_image, cmap='gray')
    # axs[1].axis('off')
    # # axs[1].set_title("source hsv_image")
    #
    # axs[2].imshow(small_mask_removed, cmap='gray')
    # axs[2].axis('off')
    # # axs[2].set_title("Small Mask Removed")
    #
    # axs[3].imshow(dilated_img, cmap='gray')
    # axs[3].axis('off')
    # # axs[3].set_title("Dilated Image")
    #
    # axs[4].imshow(eroded_img, cmap='gray')
    # axs[4].axis('off')
    # # axs[4].set_title("Eroded Image")
    #
    # plt.show()