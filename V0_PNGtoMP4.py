import cv2
import os

def images_to_video(image_folder, output_video_file, fps=30):
    """
    将文件夹内的图片转换为视频
    :param image_folder: 图片文件夹路径
    :param output_video_file: 输出视频的文件名 (例如 'output.mp4')
    :param fps: 帧率 (Frames Per Second)，你要求的是 30Hz
    """
    
    # 1. 获取文件夹内所有图片文件
    # 这里假设图片格式为 jpg 或 png，你可以根据需要添加其他格式
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    
    # 2. 对文件名进行排序
    # 注意：如果你的文件名是 frame_1.jpg, frame_10.jpg, frame_2.jpg，
    # 默认的 sort() 会导致顺序错误 (1, 10, 2)。
    # 如果文件名包含数字索引，建议确保它们是补零的 (如 001, 002) 或者使用更复杂的排序逻辑。
    images.sort() 
    
    if not images:
        print("错误：文件夹内没有找到图片。")
        return

    # 3. 读取第一张图片来获取尺寸 (视频的宽和高必须固定)
    frame_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(frame_path)
    height, width, layers = frame.shape
    size = (width, height)
    
    print(f"检测到图片尺寸: {width}x{height}, 总帧数: {len(images)}")

    # 4. 设置视频编码器
    # 'mp4v' 是 mp4 的通用编码，兼容性较好
    # 如果你在 macOS 上，也可以尝试 'avc1'
    # 注意：mp4v是有损压缩，可能导致图像质量下降
    # 如果需要无损，可以使用 'FFV1' 编码 + .avi 格式
    # fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # 无损编码
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    # 创建 VideoWriter 对象
    video = cv2.VideoWriter(output_video_file, fourcc, fps, size)
    
    # 5. 逐帧写入视频
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)
        
        # 确保读取成功且尺寸一致（如果尺寸不一致会导致写入失败）
        if img is None:
            print(f"跳过无法读取的图片: {image_name}")
            continue
        
        # 如果后续图片尺寸与第一张不同，需要 resize (可选，视你的数据情况而定)
        # img = cv2.resize(img, size) 
        
        video.write(img)

    # 6. 释放资源
    video.release()
    cv2.destroyAllWindows()
    print(f"视频已保存为: {output_video_file}")

# --- 使用示例 ---
# 请修改下面的路径为你实际的文件夹路径
image_folder_path = 'Vptac_shape_data_20260120_162028' 
output_file_name = 'output_video.mp4'
fps = 30 # 30Hz 采样率

images_to_video(image_folder_path, output_file_name, fps)