from utils.registration.registration_tree import RegistrationQuadTree, Point
#from registration_tree import RegistrationQuadTree, Point
from pathlib import Path
import numpy as np
import openslide
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import minimize
import probreg
from PIL import Image
# 设置更高的像素限制

def get_img_in_source(x: int, y: int, w: int, h: int,
                      target: openslide.OpenSlide, qtree: RegistrationQuadTree):
    '''get the corresponding img of source slide from target slide

    x,y: coors in source
    tf_param: affine mapping source to target
    '''
  #  print(str(x)+'data'+str(y))
   # print(y)
    # pts1 = tf_param.transform([[x, y],
    #                            [x + w, y],
    #                            [x + w, y + h],
    #                            [x, y + h]]).round(0).astype(int)
    #print('pts:')
    pts1 = np.float32([i[:2] for i in qtree.transform_boxes([[x, y, 1, 1],
                                                             [x + w, y, 1, 1],
                                                             [x + w, y + h, 1, 1],
                                                             [x, y + h, 1, 1]])]).round(0).astype(int)

    pts2 = np.float32([[0, 0], [w, 0], [w, w], [0, w]])

    loc = pts1.min(0)
    size = pts1.max(0) - loc
    pts1 = pts1 - loc

    if any(size > (w * 2, h * 2)):
        raise NotImplementedError('Huge gap in size between the two images')

    img = np.array(target.read_region(location=loc, size=size, level=0))

    matrix = cv2.getPerspectiveTransform(pts1.astype(np.float32), pts2.astype(np.float32))
    t_img = cv2.warpPerspective(img, matrix, (w, h))

    return t_img


def get_tf_param(x, y, qtree, max_depth=100):
    point = Point(x, y)
    if qtree.nw is not None and qtree.nw.source_boundary.contains(point) and qtree.nw.depth <= max_depth:
        tf_param = get_tf_param(x, y, qtree.nw, max_depth)
    elif qtree.ne is not None and qtree.ne.source_boundary.contains(point) and qtree.ne.depth <= max_depth:
        tf_param = get_tf_param(x, y, qtree.ne, max_depth)
    elif qtree.se is not None and qtree.se.source_boundary.contains(point) and qtree.se.depth <= max_depth:
        tf_param = get_tf_param(x, y, qtree.se, max_depth)
    elif qtree.sw is not None and qtree.sw.source_boundary.contains(point) and qtree.sw.depth <= max_depth:
        tf_param = get_tf_param(x, y, qtree.sw, max_depth)
    else:
        tf_param = qtree.tf_param

    return tf_param


def transform_min_corr(other_img, he_img):
    # 加载两张图片
    image1 = cv2.cvtColor(other_img, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(he_img, cv2.COLOR_BGR2GRAY)
    assert image1.shape == image2.shape

    # 定义目标函数：计算负的相关系数
    def objective_function(params):
        M = np.float32([[params[0], params[1], params[2]], [params[3], params[4], params[5]]])
        transformed_image = cv2.warpAffine(image1, M, (image2.shape[1], image2.shape[0]))
        non_border = transformed_image > 0
        correlation = np.abs(np.corrcoef(transformed_image[non_border].flatten(), image2[non_border].flatten())[0, 1])
        return -correlation

    initial_params = [1, 0, 0, 0, 1, 0]
    result = minimize(objective_function, initial_params, method='Nelder-Mead')

    # # 打印最终找到的参数和相关系数
    # print("Optimized Parameters:", result.x)
    # print("Maximized r:", -result.fun)

    # 使用找到的参数进行仿射变换
    M = np.float32([[result.x[0], result.x[1], result.x[2]], [result.x[3], result.x[4], result.x[5]]])
    tf_other_img = cv2.warpAffine(other_img, M, (image2.shape[1], image2.shape[0]))
    return M, tf_other_img


# # 约束只允许平移、旋转和一定范围内的缩放
# def transform_min_corr(other_img, he_img):
#     # 加载两张图片并转换为灰度图
#     image1 = cv2.cvtColor(other_img, cv2.COLOR_BGR2GRAY)
#     image2 = cv2.cvtColor(he_img, cv2.COLOR_BGR2GRAY)
#     assert image1.shape == image2.shape
#
#     def objective_function(params):
#         angle = params[0]
#         tx = params[1]
#         ty = params[2]
#         scale = params[3]
#
#         scale = max(0.8, min(1.2, scale))
#
#         M = cv2.getRotationMatrix2D((image2.shape[1] // 2, image2.shape[0] // 2), angle, scale)
#         M[0, 2] += tx
#         M[1, 2] += ty
#
#         # 应用仿射变换
#         transformed_image = cv2.warpAffine(image1, M, (image2.shape[1], image2.shape[0]))
#         non_border = transformed_image > 0
#         correlation = np.abs(np.corrcoef(transformed_image[non_border].flatten(), image2[non_border].flatten())[0, 1])
#
#         return -correlation
#
#     # 初始参数：角度、x轴平移、y轴平移、缩放因子
#     initial_params = [0, 0, 0, 1]  # 初始旋转角度为0，无平移，缩放因子为1
#
#     # 使用Nelder-Mead方法最小化目标函数
#     result = minimize(objective_function, initial_params, method='Nelder-Mead')
#
#     # 使用找到的参数进行仿射变换
#     optimized_scale = max(0.8, min(1.2, result.x[3]))  # 确保缩放因子在限制范围内
#     M = cv2.getRotationMatrix2D((image2.shape[1] // 2, image2.shape[0] // 2), result.x[0], optimized_scale)
#     M[0, 2] += result.x[1]
#     M[1, 2] += result.x[2]
#     tf_other_img = cv2.warpAffine(other_img, M, (image2.shape[1], image2.shape[0]))
#
#     return M, tf_other_img


if __name__ == '__main__':
    parameters = {
        # feature extractor parameters
        "point_extractor": "sift",  # orb , sift
        "maxFeatures": 4096,
        "crossCheck": False,
        "flann": False,
        "ratio": 0.7,
        "use_gray": True,

        # QTree parameter
        "homography": True,
        "filter_outliner": False,
        "debug": True,
        "target_depth": 1,
        "run_async": True,
        "thumbnail_size": (2048, 2048)
    }

    # source = Path("data/CRC-A1-1.tif")
    # target = Path("data/CRC-A1-1 HE.tif")
    source = Path("/homellm8t/zhaoxz/WSI_test/IHC/Q1906740-1—FB227P0145-KBL2.tiff")
    target = Path("/homellm8t/zhaoxz/WSI_test/HE/Q1906740-1—FB227P0145-KBL2.tiff")
    qtree = RegistrationQuadTree(source_slide_path=source,
                                 target_slide_path=target, **parameters)

    x = 500
    y = 500
    size = (1024, 1024)
    s = openslide.open_slide(source._str)
    t = openslide.open_slide(target._str)
    th_s = np.array(s.get_thumbnail(size))
    th_t = np.array(t.get_thumbnail(size))
    box = [x, y, *size]
    tbox = qtree.transform_boxes([box])[0].round(0).astype(int)
    scale = max(th_s.shape) / max(s.level_dimensions[0])
    w = 256
    x1, x2, y1, y2 = 500, 500 + w, 500, 500 + w

    ## detail
    ## based on multi-scale affine transform
    # pts1 = qtree.transform_boxes(np.array([[x1, y1, 1, 1], [x2, y1, 1, 1], [x2, y2, 1, 1], [x1, y2, 1, 1]]) / scale)
    # pts1 = np.float32([(i[:2] * scale).round(0).astype(int) for i in pts1])
    # pts2 = np.float32([[0, 0], [w, 0], [w, w], [0, w]])
    # a = th_s[y1:y2, x1:x2]
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # b = cv2.warpPerspective(th_t, matrix, (x2 - x1, y2 - y1))
    #
    # a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    # b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    # _, a = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # fig, ax = plt.subplots()
    # ax.imshow(a, cmap='Reds', alpha=0.5)
    # ax.imshow(b, cmap='Blues', alpha=0.5)
    # fig.show()

    # ## based on single scale affine transform
    # M = np.zeros([2, 3])
    # M[:2, :2] = qtree.tf_param.b
    # M[:, 2] = qtree.tf_param.t * scale
    # th_s_ = cv2.warpAffine(th_s, M=M, dsize=th_t.shape[:2])
    # a = th_s_[y1:y2, x1:x2]
    # b = th_t[y1:y2, x1:x2]
    # a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    # b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    # _, a = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # fig, ax = plt.subplots()
    # ax.imshow(a, cmap='Reds', alpha=0.5)
    # ax.imshow(b, cmap='Blues', alpha=0.5)
    # fig.show()

    ## coords transfrom
    x, y = 12000, 2000
    w, h = 8096, 8096
    a = np.array(s.read_region(location=(x, y), size=(w, h), level=0))
    b = get_img_in_source(x=x, y=y, w=w, h=h, target=t, qtree=qtree)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    _, a = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.figure()
    fig, ax = plt.subplots()
    ax.imshow(a, cmap='Reds', alpha=0.5)
    ax.imshow(b, cmap='Blues', alpha=0.5)
    fig.show()

    # ## coords transfrom
    # x, y = 12000, 2000
    # w, h = 2048, 2048
    # tf_param = get_tf_param(x, y, qtree, max_depth=100)
    # a = np.array(s.read_region(location=(x, y), size=(w, h), level=0))
    # b = get_img_in_source(x=x, y=y, w=w, h=h, tf_param=tf_param, target=t, qtree=qtree)
    # a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    # b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    # _, a = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plt.figure()
    # fig, ax = plt.subplots()
    # ax.imshow(a, cmap='Reds', alpha=0.5)
    # ax.imshow(b, cmap='Blues', alpha=0.5)
    # fig.show()

    # ## warp whole image first
    # x, y = 2116, 2605
    # w, h = 512, 512
    # M = np.zeros([2, 3])
    # M[:2, :2] = qtree.tf_param.b
    # M[:, 2] = qtree.tf_param.t
    # imgs = np.array(s.get_thumbnail(s.level_dimensions[0]))
    # imgt = np.array(t.get_thumbnail(t.level_dimensions[0]))
    #
    # imgs = cv2.warpAffine(imgs, M=M, dsize=t.level_dimensions[0])
    # a = imgs[y:y + h, x:x + w]
    # b = imgt[y:y + h, x:x + w]
    #
    # a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    # b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    # _, a = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # fig, ax = plt.subplots()
    # ax.imshow(a, cmap='Reds', alpha=0.5)
    # ax.imshow(b, cmap='Blues', alpha=0.5)
    # fig.show()
