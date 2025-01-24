import os
import cv2
import natsort

def add_camera_label(image, camera_name):
    """在图像上添加相机名称标签，使用更美观的样式"""
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(camera_name, font, 0.8, 2)[0]
    
    # 创建半透明背景
    padding = 10
    overlay = image.copy()
    cv2.rectangle(overlay, 
                 (0, 0), 
                 (text_size[0] + 2*padding, text_size[1] + 2*padding), 
                 (47, 47, 47), 
                 -1)
    
    # 应用半透明效果
    alpha = 0.7
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # 添加文本
    position = (padding, text_size[1] + padding//2)
    cv2.putText(image, camera_name, position, font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def safe_read_resize(path, size=(1056, 696), border_width=5):
    """安全读取并调整图像大小，使用灰色边框替代黑色边框"""
    img = cv2.imread(path)
    if img is None:
        return None
    
    # 获取原始图像高度和宽度
    h, w = img.shape[:2]
    target_h, target_w = size
    
    # 扣除边框后的可用尺寸
    available_w = target_w - 2 * border_width
    available_h = target_h - 2 * border_width
    
    # 计算缩放比例
    scale = min(available_w / w, available_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 调整图像大小
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 计算需要添加的边框
    top = border_width + (available_h - new_h) // 2
    bottom = border_width + (available_h - new_h) - (top - border_width)
    left = border_width + (available_w - new_w) // 2
    right = border_width + (available_w - new_w) - (left - border_width)
    
    # 添加深灰色边框
    bordered_img = cv2.copyMakeBorder(
        resized_img, 
        top, bottom, left, right, 
        cv2.BORDER_CONSTANT, 
        value=[47, 47, 47]  # 深灰色背景
    )
    
    return bordered_img

def process_timestamp_images(visual_dir, tmp_t_dir, output_dir):
    """处理并拼接图像"""
    os.makedirs(output_dir, exist_ok=True)
    timestamps = natsort.natsorted([
        d for d in os.listdir(visual_dir) 
        if os.path.isdir(os.path.join(visual_dir, d))
    ])

    # 定义统一的图像尺寸
    ORIG_SIZE = (480, 360)  # 原始相机图像
    FRONT_REAR_SIZE = (480, 360)  # 前后视角
    TOP_VIEW_SIZE = (480, 480)  # 俯视图

    for timestamp in timestamps:
        # 构建路径
        orig_path = os.path.join(visual_dir, timestamp)
        vis_path = os.path.join(tmp_t_dir, timestamp)

        # 读取原始图像
        orig_imgs = [
            safe_read_resize(os.path.join(orig_path, f'{i}.jpg'), size=ORIG_SIZE) 
            for i in range(4)
        ]

        # 读取前后视角可视化图像
        vis_imgs2 = [
            safe_read_resize((vis_path + f'_{view}.png'), size=FRONT_REAR_SIZE) 
            for view in ['front_view', 'rear_view']
        ]

        # 读取俯瞰可视化图像
        vis_imgs1 = [
            safe_read_resize((vis_path + f'_{view}.png'), size=TOP_VIEW_SIZE) 
            for view in ['45_degree_view', 'top_view']
        ]

        # 检查图像是否全部成功读取
        if any(img is None for img in orig_imgs + vis_imgs1 + vis_imgs2):
            print(f"跳过 {timestamp}: 存在无法读取的图像")
            continue

        # 添加相机标签
        camera_names = ['Front', 'Left', 'Right', 'Back']
        orig_imgs_labeled = [add_camera_label(img, name) for img, name in zip(orig_imgs, camera_names)]

        # 在原始图像之间添加边框
        border = 5
        orig_imgs_with_border = []
        for i, img in enumerate(orig_imgs_labeled):
            if i < len(orig_imgs_labeled) - 1:
                img = cv2.copyMakeBorder(img, 0, 0, 0, border, cv2.BORDER_CONSTANT, value=[47, 47, 47])
            orig_imgs_with_border.append(img)

        # 横向拼接原始图像
        orig_img_concat = cv2.hconcat(orig_imgs_with_border)

        # 前后视角拼接
        vis_imgs2_with_border = [vis_imgs2[0]]
        vis_imgs2_with_border.append(
            cv2.copyMakeBorder(vis_imgs2[1], 0, 0, border, 0, cv2.BORDER_CONSTANT, value=[47, 47, 47])
        )
        front_rear_concat = cv2.hconcat(vis_imgs2_with_border)

        # 确保第一行图像高度一致
        first_row_height = max(orig_img_concat.shape[0], front_rear_concat.shape[0])
        if orig_img_concat.shape[0] < first_row_height:
            orig_img_concat = cv2.copyMakeBorder(
                orig_img_concat, 
                0, first_row_height - orig_img_concat.shape[0], 0, 0, 
                cv2.BORDER_CONSTANT, value=[47, 47, 47]
            )
        if front_rear_concat.shape[0] < first_row_height:
            front_rear_concat = cv2.copyMakeBorder(
                front_rear_concat, 
                0, first_row_height - front_rear_concat.shape[0], 0, 0, 
                cv2.BORDER_CONSTANT, value=[47, 47, 47]
            )

        # 第一行：原始图像 + 前后视角，确保宽度一致
        first_row = cv2.hconcat([orig_img_concat, front_rear_concat])

        # 俯瞰拼接可视化图
        vis_imgs1_with_border = [vis_imgs1[0]]
        vis_imgs1_with_border.append(
            cv2.copyMakeBorder(vis_imgs1[1], 0, 0, border, 0, cv2.BORDER_CONSTANT, value=[47, 47, 47])
        )
        top_view_concat = cv2.hconcat(vis_imgs1_with_border)

        # 确保总宽度一致
        if top_view_concat.shape[1] < first_row.shape[1]:
            # 如果俯视图总宽度小于第一行，添加边框使其相等
            padding = (first_row.shape[1] - top_view_concat.shape[1]) // 2
            top_view_concat = cv2.copyMakeBorder(
                top_view_concat,
                0, 0,
                padding, first_row.shape[1] - top_view_concat.shape[1] - padding,
                cv2.BORDER_CONSTANT, value=[47, 47, 47]
            )
        elif top_view_concat.shape[1] > first_row.shape[1]:
            # 如果俯视图总宽度大于第一行，添加边框使第一行相等
            padding = (top_view_concat.shape[1] - first_row.shape[1]) // 2
            first_row = cv2.copyMakeBorder(
                first_row,
                0, 0,
                padding, top_view_concat.shape[1] - first_row.shape[1] - padding,
                cv2.BORDER_CONSTANT, value=[47, 47, 47]
            )

        # 在两行之间添加边框
        first_row = cv2.copyMakeBorder(first_row, 0, border, 0, 0, cv2.BORDER_CONSTANT, value=[47, 47, 47])

        # 垂直拼接最终图像
        final_image = cv2.vconcat([first_row, top_view_concat])

        # 保存结果
        output_path = os.path.join(output_dir, f'{timestamp}_combined.png')
        cv2.imwrite(output_path, final_image)

        print(f"处理完成: {timestamp}")

# 设置路径
visual_dir = '/home/tianws/keyidel/vis_occ/visual_dir_val'
tmp_t_dir = '/home/tianws/keyidel/vis_occ/ori_code/tmp_t'
output_dir = '/home/tianws/keyidel/vis_occ/ori_code/combined_images'

# 执行处理
process_timestamp_images(visual_dir, tmp_t_dir, output_dir)