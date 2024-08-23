import numpy as np
import cv2
import openslide
from PIL import Image
import tifffile as tiff
import os
import pyvips
import json
from multiprocessing import Pool 
# save the matrix to tiff img #
def save_large_matrix_as_tiff(matrix, filename):
    height, width, channels = matrix.shape
    
    # 创建一个pyvips图像对象
    image = pyvips.Image.new_from_memory(matrix.tobytes(), width, height, channels, 'uchar')
    
    # 保存为TIFF文件
    image.tiffsave(filename, tile=True, pyramid=True, compression='lzw', bigtiff=True)
    
def create_thumbnail(filename, thumbnail_filename, width=1024):
    image = pyvips.Image.new_from_file(filename, access='sequential')
    thumbnail = image.thumbnail_image(width)
    thumbnail.write_to_file(thumbnail_filename)
        
def process_and_save_image(image_dir, adjust_params, save_filename,save_thu_filename,chunk_size=1024):
    def adjust_black_white(image, black_level, white_level):
        black_level = np.clip(black_level, 0, 255)
        white_level = np.clip(white_level, 0, 255)
        normalized_img = image / 255.0
        adjusted_img = (normalized_img - black_level / 255.0) / (white_level / 255.0 - black_level / 255.0)
        adjusted_img = np.clip(adjusted_img, 0, 1) * 255
        return adjusted_img.astype(np.uint8)

    def adjust_gamma(image, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def process_image(file_path, black_level, white_level, gamma, change_color, chunk_size):
        slide = openslide.OpenSlide(file_path)
        width, height = slide.level_dimensions[0]
        combined_image = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):
                region = slide.read_region((x, y), 0, (min(chunk_size, width - x), min(chunk_size, height - y)))
                image = np.array(region.convert('L'))
                adjusted_image = adjust_black_white(image, black_level, white_level)
                gamma_adjusted_image = adjust_gamma(adjusted_image, gamma)

                bw_array = np.array(gamma_adjusted_image)
                blue_black_array = np.zeros((bw_array.shape[0], bw_array.shape[1], 3), dtype=np.uint8)

                if change_color == 'SpRed':
                    blue_black_array[:, :, 0] = bw_array
                elif change_color == 'SpGreen':
                    blue_black_array[:, :, 1] = bw_array
                    blue_black_array[:, :, 2] = (bw_array * 126 / 255).astype(np.uint8)
                elif change_color == 'AQUA':
                    blue_black_array[:, :, 2] = bw_array
                elif change_color == 'Cy5.5':
                    blue_black_array[:, :, 0] = bw_array
                    blue_black_array[:, :, 2] = bw_array
                elif change_color == '350x431M':
                    blue_black_array[:, :, 1] = bw_array
                    blue_black_array[:, :, 2] = bw_array
                elif change_color == 'SpGold':
                    blue_black_array[:, :, 0] = bw_array
                    blue_black_array[:, :, 1] = bw_array

                combined_image[y:y+chunk_size, x:x+chunk_size] += blue_black_array[:min(chunk_size, height - y), :min(chunk_size, width - x)]

        return combined_image

    combined_image = None
    
    ## {'AQUA': (0, 142, 1), 'SpGold': (9, 56, 1), 'SpRed': (10, 95, 1), 'Cy5.5': (20, 206, 1), 'SpGreen': (19, 101, 1), '350x431M': (24, 255, 1)}
    color_types = ['AQUA', 'SpGold', 'SpRed', 'Cy5.5', 'SpGreen', '350X431M']
    image_files = [f for f in os.listdir(image_dir) if any(color in f for color in color_types)]

    for color_type, (black_level, white_level, gamma) in zip(color_types, adjust_params.values()):
        file_path = next((os.path.join(image_dir, f) for f in image_files if color_type in f), None)
        if file_path:
            print(f"Processing {color_type}")
            rgb_image = process_image(file_path, black_level, white_level, gamma, color_type, chunk_size)
            if combined_image is None:
                combined_image = rgb_image
            else:
                combined_image = cv2.addWeighted(combined_image, 1, rgb_image, 1, 0)

    # 转换回PIL图像
    save_large_matrix_as_tiff(combined_image,save_filename)
    create_thumbnail(save_filename,save_thu_filename)
    print(f"Processing completed and saved as '{save_filename}'.")
def find_and_convert_sample(json_file_path, sample_id):
    # 打开并读取UTF-8编码的JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 查找特定ID的样本
    def find_sample_by_id(sample_id):
        for item in data:
            if item['id'] == sample_id:
                return item
        return None
    # 转换样本数据为所需格式
    def convert_sample_data(sample):
        if sample:
            return {entry['name']: (entry['value1'], entry['value2'], entry['value3']) for entry in sample['data']}
        return {}

    # 查找并转换数据
    sample = find_sample_by_id(sample_id)
    if sample:
        converted_data = convert_sample_data(sample)
        return converted_data
    else:
        return []
def get_color_from_josn(json_path,sample_id):
    sample_id=sample_id.split('/')[-1].split('_')[0]
    result_color_code=find_and_convert_sample(json_path,sample_id)
    return result_color_code

def process_image_main(args):
    i, image_dir, json_dir, save_dir = args
    sample_dir = os.path.join(image_dir, i)
    save_file_name = os.path.join(save_dir, str(i.split('_')[0]) + '.tiff')
    save_thu_file_name = os.path.join(save_dir, str(i.split('_')[0]) + '.jpg')
    adj_color_code = get_color_from_josn(json_dir, i)
    process_and_save_image(sample_dir, adj_color_code, save_file_name, save_thu_file_name, chunk_size=1024)

def color_set(image_dir, json_dir, save_dir):
    sample_info = os.listdir(image_dir)
    args = [(i, image_dir, json_dir, save_dir) for i in sample_info]
    with Pool(processes=3) as pool:
        pool.map(process_image_main, args)

# example

#
image_dir = "/homellm8t/zhaoxz/gray/"
json_dir='/homellm8t/zhaoxz/script/color_default.json'
save_dir='/homellm8t/zhaoxz/IHC_merge'
color_set(image_dir,json_dir,save_dir)
