import cv2 #opencv读取的格式是BGR格式
import matplotlib.pyplot as plt
import numpy as np
import os
import json

plate_type = input("请输入车牌类型（7/8）, 7表示普通蓝牌，8表示新能源车牌：")
temp_json = "03.json"
temp_jpg = "03.jpg"

img = cv2.imread(temp_jpg)
h, w, c = img.shape

def cv_show(name, img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def get_point(point):
    x = int(point[0])
    y = int(point[1])
    return (x,y)
    
src_list = [get_point(point1), get_point(point2), get_point(point3), get_point(point4)]
for i, pt in enumerate(src_list):
    cv2.circle(img, pt, 5, (0, 0, 255), -1)
    cv2.putText(img,str(i+1),(pt[0]+5,pt[1]+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
pts1 = np.float32(src_list)

# 透视变换
pts2 = np.float32([[0, 0], [0, w - 2], [h - 2, w - 2], [h - 2, 0]])
#计算变换矩阵
matrix = cv2.getPerspectiveTransform(pts1, pts2)
#应用变换矩阵
result = cv2.warpPerspective(Original, matrix, (h, w))

plt.figure(figsize=(10, 8), dpi=200)
plt.imshow(img)

plt.imshow(result)

plate = cv2.resize(result, (480, 140))

# 标准车牌高为140mm，宽为440mm车牌的宽高比为3.14。

plt.imshow(plate)

gauss_plate = cv2.GaussianBlur(plate, (3,3), 0)

plt.imshow(gauss_plate)

gray_image = cv2.cvtColor(gauss_plate, cv2.COLOR_RGB2GRAY)

plt.imshow(gray_image,cmap='gray')

image = gray_image.copy()

rows = image.shape[0]
cols = image.shape[1]
image.shape

gray_mean = np.mean(image)*0.9
gray_mean

for row in range(rows):
    res = 0
    for col in range(cols):
        if image[row][col] > gray_mean:
            image[row][col] = 255
        else:
            image[row][col] = 0


plt.imshow(image,cmap='gray')

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
plt.imshow(imageh,cmap='gray')

imagev = imageh.copy()

rows = imagev.shape[0]
cols = imagev.shape[1]

hdv = []
for col in range(cols):
    res = 0
    for row in range(rows):
        if imagev[row][col] == 0:
            res = res+1
    hdv.append(res)
len(hdv)

x = [x for x in range(cols)]
y = hdv
plt.bar(x,y,color='black',width=1)


mean = sum(hdv)/len(hdv)
mean


# In[30]:


for i in range(cols):
    if hdv[i] < mean/2:
        hdv[i] = 0


# In[31]:


x = [x for x in range(cols)]
y = hdv
plt.bar(x,y,color='black',width=1)


# In[32]:


regionx = []
for i in range(0, cols-1):
    if hdv[i] == 0 and hdv[i+1] != 0:
        regionx.append(i)
        break
        
for i in range(cols-1, 0, -1):
    if hdv[i] == 0 and hdv[i-1] != 0:
        regionx.append(i)
        break


# In[33]:


imagev = imageh[:,regionx[0]:regionx[1]]
plt.imshow(imagev,cmap='gray')


# In[34]:


image = cv2.resize(imagev, (449, 90))
plt.imshow(image,cmap='gray')
plt.show()


# In[35]:


image = 255 - image


# In[36]:


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
    for i in range(7):
        plt.subplot(1, 7, i+1), plt.imshow(plate[i], 'gray')
    plt.show()
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
    for i in range(8):
        plt.subplot(1, 8, i+1), plt.imshow(plate[i], 'gray')
    plt.show()




# reader = easyocr.Reader(['en']) # 只需要运行一次就可以将模型加载到内存中
# result = reader.readtext(image6)
# result

# In[38]:


# 模版匹配
# 准备模板(template[0-9]为数字模板；)
template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '云', '京', '冀', '吉', '宁', '川', '新', '晋', '桂', '沪', '津', '浙', '渝', '湘', '琼', '甘', '皖', '粤', '苏', '蒙', '藏', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑']


# 读取文件夹中的所有图片，返回图片路径列表
def read_directory(directory_name):
    return [directory_name + "/" + filename for filename in os.listdir(directory_name)]

# 获得中文模板列表（匹配车牌的第一个字符）
def get_chinese_words_list():
    return [read_directory('./char/' + template[i]) for i in range(34, 65)]

# 获得英文模板列表（匹配车牌的第二个字符）
def get_eng_words_list():
    return [read_directory('./char/' + template[i]) for i in range(10, 34)]

# 获得英文和数字模板列表（匹配车牌后面的字符）
def get_eng_num_words_list():
    return [read_directory('./char/' + template[i]) for i in range(0, 34)]

# 模板匹配得分
def template_score(template, image):
    template_img = cv2.imdecode(np.fromfile(template, dtype=np.uint8), 1)
    template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
    _, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
    
    image_ = image.copy()
    height, width = image_.shape
    template_img = cv2.resize(template_img, (width, height))
    
    result = cv2.matchTemplate(image_, template_img, cv2.TM_CCOEFF)
    return result[0][0]

# 对分割得到的字符逐一进行模板匹配
def template_matching(word_images):
    results = []
    
    for index, word_image in enumerate(word_images):
        if index == 0:  # 匹配第一个字符（中文）
            best_score = [max([template_score(c_word, word_image) for c_word in chinese_words]) for chinese_words in chinese_words_list]
            i = best_score.index(max(best_score))
            results.append(template[34 + i])
        elif index == 1:  # 匹配第二个字符（英文）
            best_score = [max([template_score(e_word, word_image) for e_word in eng_word_list]) for eng_word_list in eng_words_list]
            i = best_score.index(max(best_score))
            results.append(template[10 + i])
        else:  # 匹配后续字符（英文或数字）
            best_score = [max([template_score(e_num_word, word_image) for e_num_word in eng_num_word_list]) for eng_num_word_list in eng_num_words_list]
            i = best_score.index(max(best_score))
            results.append(template[i])
    
    return results

# 更正字符'0'和'D'
def correct_characters(result, plate):
    for i in range(len(result)):
        if result[i] == '0':
            res = sum([1 for j in range(10) for k in range(46) if plate[i][j][k] == 255])
            if res > 265:
                result[i] = 'D'
        elif result[i] == 'D':
            res = sum([1 for j in range(10) for k in range(46) if plate[i][j][k] == 255])
            if res < 265:
                result[i] = '0'
    return result

# 获取模板列表
chinese_words_list = get_chinese_words_list()
eng_words_list = get_eng_words_list()
eng_num_words_list = get_eng_num_words_list()

# 假设`plate`已定义并且包含字符图像
word_images_ = plate.copy()

# 获取匹配结果
result = template_matching(word_images_)

# 输出车牌号
print("".join(result))

# 针对'0'与'D'的更正
result = correct_characters(result, plate)

# 输出最终车牌号
print("".join(result))


