# 图像读取及车牌识别
import cv2  # OpenCV库，用于图像处理
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# 询问用户车牌类型
plate_type = input("请输入车牌类型（7/8）, 7表示普通蓝牌，8表示新能源车牌：")

# 设置文件路径
temp_json = "01.json"
temp_jpg = "01.jpg"

# 读取原始图像
img = cv2.imread(temp_jpg)
h, w, c = img.shape

# 显示图像的函数
def cv_show(name, img):
    """
    显示图像函数
    :param name: 窗口名称
    :param img: 要显示的图像
    """
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 读取原始图像
Original = cv2.imread(temp_jpg)

# 读取标注信息 (json)
with open(temp_json, 'r') as file:
    data = json.load(file)
    # 提取多边形的四个点（逆时针顺序）
    polygon_points = data["shapes"][0]["points"]
    
    # 根据逆时针顺序：左上、右上、右下、左下
    # 首先按照 y 坐标排序分成上下两组点
    sorted_by_y = sorted(polygon_points, key=lambda p: p[1])
    top_points = sorted_by_y[:2]
    bottom_points = sorted_by_y[2:]
    
    # 然后在每组中按 x 坐标排序
    top_points = sorted(top_points, key=lambda p: p[0])    # x坐标小的在前
    bottom_points = sorted(bottom_points, key=lambda p: p[0], reverse=True)  # x坐标大的在前
    
    point1 = top_points[0]      # 左上
    point4 = top_points[1]      # 右上
    point3 = bottom_points[0]   # 右下
    point2 = bottom_points[1]   # 左下

# 获取坐标点
def get_point(point):
    """
    将点的坐标转换为整数
    :param point: 输入的点 (x, y)
    :return: 返回整数格式的坐标 (x, y)
    """
    x = int(point[0])
    y = int(point[1])
    return (x, y)

# 获取标注框的四个顶点坐标
src_list = [get_point(point1), get_point(point2), get_point(point3), get_point(point4)]
for i, pt in enumerate(src_list):
    cv2.circle(img, pt, 5, (0, 0, 255), -1)
    cv2.putText(img, str(i + 1), (pt[0] + 5, pt[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# 透视变换源点
pts1 = np.float32(src_list)

pts2 = np.float32([[0, 0], [0, w - 2], [h - 2, w - 2], [h - 2, 0]])

# 计算变换矩阵
matrix = cv2.getPerspectiveTransform(pts1, pts2)
# 应用变换矩阵，使用目标尺寸
result = cv2.warpPerspective(Original, matrix, (h, w))

# 显示原图和透视变换后的图像
plt.figure(figsize=(10, 8), dpi=200)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 显示标注框图像
plt.show()

# 裁剪车牌区域
plate = result  # 直接使用透视变换后的图像
# plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
# plt.show()

# 高斯模糊处理
gauss_plate = cv2.GaussianBlur(plate, (3, 3), 0)
# plt.imshow(cv2.cvtColor(gauss_plate, cv2.COLOR_BGR2RGB))
# plt.show()

# 转为灰度图
gray_image = cv2.cvtColor(gauss_plate, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray_image, cmap='gray')
# plt.show()

# 复制灰度图并进行二值化处理
image = gray_image.copy()

# 获取图像的尺寸
rows, cols = image.shape

# 计算图像的均值，用于阈值化
gray_mean = np.mean(image) * 0.9

# 基于均值进行二值化
for row in range(rows):
    for col in range(cols):
        if image[row][col] > gray_mean:
            image[row][col] = 255
        else:
            image[row][col] = 0

# 显示二值化后的图像
# plt.imshow(image, cmap='gray')
# plt.show()

# 水平直方图，统计每一行黑色像素数量
hd = []
for row in range(rows):
    res = 0
    for col in range(cols):
        if image[row][col] == 0:
            res += 1
    hd.append(res)

# 查找车牌区域的上下边界
mean = sum(hd) / len(hd) * 0.3  # 调整阈值
region = []

# 从上到下扫描
for i in range(rows):
    if hd[i] > mean:
        region.append(i)
        break

# 从下到上扫描
for i in range(rows-1, -1, -1):
    if hd[i] > mean:
        region.append(i)
        break

# 确保找到了边界
if len(region) < 2:
    print("警告：未能检测到有效的上下边界，使用整个图像高度")
    region = [0, rows-1]

# 裁剪出车牌区域
imageh = image[region[0]:region[1], :]
# plt.imshow(imageh, cmap='gray')
# plt.show()

# 纵向直方图，统计每一列黑色像素数量
imagev = imageh.copy()
rows, cols = imagev.shape
hdv = []
for col in range(cols):
    res = 0
    for row in range(rows):
        if imagev[row][col] == 0:
            res += 1
    hdv.append(res)

# 绘制纵向直方图
x = [x for x in range(cols)]
y = hdv
# plt.bar(x, y, color='black', width=1)
# plt.show()

# 计算纵向均值，并筛选出车牌字符的区域
mean = sum(hdv) / len(hdv)
for i in range(cols):
    if hdv[i] < mean / 2:
        hdv[i] = 0

# 绘制更新后的纵向直方图
x = [x for x in range(cols)]
y = hdv
# plt.bar(x, y, color='black', width=1)
# plt.show()

# 查找车牌字符区域的左右边界
if plate_type == "8":
    # 八位车牌使用完整宽度
    regionx = [0, cols-1]
else:
    # 七位车牌使用原有的边界检测逻辑
    regionx = []
    found_start = False
    found_end = False

    # 从左向右查找起始点
    for i in range(0, cols - 1):
        if hdv[i] == 0 and hdv[i + 1] != 0:
            regionx.append(i)
            found_start = True
            break

    # 从右向左查找终止点
    for i in range(cols - 1, 0, -1):
        if hdv[i] == 0 and hdv[i - 1] != 0:
            regionx.append(i)
            found_end = True
            break

    # 如果没有找到有效的边界，使用默认值
    if not found_start or not found_end or len(regionx) < 2:
        print("警告：未能准确定位字符边界，使用默认值")
        regionx = [0, cols-1]

# 裁剪出车牌字符区域
imagev = imageh[:, regionx[0]:regionx[1]]
# plt.imshow(imagev, cmap='gray')
# plt.show()

# 调整图像大小以适配车牌字符的大小
image = cv2.resize(imagev, (440, 90))

plt.imshow(image, cmap='gray')
plt.show()

# 将图像反转为黑底白字
image = 255 - image

# 根据车牌类型选择分割方案
if plate_type == "7":
    # 七位蓝牌车牌分割方案
    image1 = image[:, 7:62]      # 省份缩写
    image2 = image[:, 67:118]    # 字母
    image3 = image[:, 152:211]   # 第3位
    image4 = image[:, 213:272]   # 第4位
    image5 = image[:, 272:331]   # 第5位
    image6 = image[:, 333:390]   # 第6位
    image7 = image[:, 390:447]   # 第7位
    plate = [image1, image2, image3, image4, image5, image6, image7]
    num_chars = 7
else:
    # 八位新能源车牌分割方案
    image1 = image[:,0:50]
    image2 = image[:,48:104]
    image3 = image[:,145:191]
    image4 = image[:,199:245]
    image5 = image[:,250:296]
    image6 = image[:,303:349]
    image7 = image[:,353:399]
    image8 = image[:,403:449]
    plate = [image1, image2, image3, image4, image5, image6, image7, image8]
    num_chars = 8

# 将分割出的字符显示出来
plt.figure(figsize=(15, 3))
for i in range(num_chars):
    plt.subplot(1, num_chars, i + 1)
    plt.imshow(plate[i], 'gray')
plt.show()

template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z',
            '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁', '蒙', '闽',
            '宁', '青', '琼', '陕', '苏', '湘', '新', '渝', '豫', '粤', '云', '浙', '藏', '川', '鄂', '甘', '赣', '贵', '皖']

# 获得模板列表
def read_directory(directory_name):
    return [directory_name + "/" + filename for filename in os.listdir(directory_name)]

def get_chinese_words_list():
    return [read_directory('./char/' + template[i]) for i in range(34, 65)]

def get_eng_words_list():
    return [read_directory('./char/' + template[i]) for i in range(10, 34)]

def get_eng_num_words_list():
    return [read_directory('./char/' + template[i]) for i in range(0, 34)]

# 模板匹配中的字符模板列表
chinese_words_list = get_chinese_words_list()  # 获取中文字符模板列表
eng_words_list = get_eng_words_list()  # 获取英文字符模板列表
eng_num_words_list = get_eng_num_words_list()  # 获取数字字符模板列表

# 模板匹配函数
def template_score(template, image):
    template_img = cv2.imdecode(np.fromfile(template, dtype=np.uint8), 1)
    template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
    ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
    image_ = cv2.resize(image.copy(), (template_img.shape[1], template_img.shape[0]))
    result = cv2.matchTemplate(image_, template_img, cv2.TM_CCOEFF)
    return result[0][0]

# 模板匹配主函数
def template_matching(word_images):
    results = []
    for index, word_image in enumerate(word_images):
        best_score = []
        if index == 0:
            word_list = chinese_words_list
            start_idx = 34
        elif index == 1:
            word_list = eng_words_list
            start_idx = 10
        else:
            word_list = eng_num_words_list
            start_idx = 0

        for word_group in word_list:
            score = [template_score(word, word_image) for word in word_group]
            best_score.append(max(score))

        i = best_score.index(max(best_score))
        results.append(template[start_idx + i])
    
    return results

# 调用函数并输出结果
result = template_matching(plate)  # 移除 num_chars 参数
print("".join(result))

# 处理字符识别中的 `0` 和 `D` 的误判问题
for i in range(len(result)):
    if result[i] == '0':
        res = np.sum(plate[i] == 255)
        if res > 265:
            result[i] = 'D'

for i in range(len(result)):
    if result[i] == 'D':
        res = np.sum(plate[i] == 255)
        if res < 265:
            result[i] = '0'

# 输出最终结果
print("Final plate recognition: " + "".join(result))


