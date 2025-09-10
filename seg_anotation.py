import json
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
from datetime import datetime

def yolov8_predict_to_coco(model_path, images_dir, output_json_path, class_names, confidence_threshold=0.5):
    """
    使用 YOLOv8 模型預測圖片並轉換為標準 COCO JSON 格式
    
    Args:
        model_path: YOLOv8 模型權重路径
        images_dir: 圖片資料夾路径
        output_json_path: 輸出的 COCO JSON 文件路径
        class_names: 類別名稱列表 ['class1', 'class2', ...]
        confidence_threshold: 信心度閾值
    """
    
    # 載入模型
    model = YOLO(model_path)
    print(f"載入模型: {model_path}")
    
    # 初始化標準 COCO 格式數據結構
    coco_data = {
        "info": {
            "description": "YOLOv8 Auto-generated COCO dataset",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "YOLOv8 Auto Annotation Tool",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 添加類別信息
    for idx, class_name in enumerate(class_names):
        coco_data["categories"].append({
            "id": idx + 1,
            "name": class_name,
            "supercategory": "object"
        })
    
    annotation_id = 1
    image_id = 1
    
    # 處理圖片資料夾中的所有圖片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
        image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
    
    print(f"找到 {len(image_files)} 張圖片")
    
    for image_path in image_files:
        print(f"處理圖片: {image_path}")
        
        # 讀取圖片獲取尺寸
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"無法讀取圖片: {image_path}")
            continue
            
        height, width = img.shape[:2]
        
        # 添加圖片信息（標準COCO格式）
        coco_data["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_path.name,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 使用模型預測
        results = model(str(image_path), conf=confidence_threshold)
        
        for result in results:
            if result.masks is not None and len(result.masks) > 0:
                masks = result.masks.xy  # 獲取多邊形座標
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for mask, cls, conf in zip(masks, classes, confidences):
                    if conf < confidence_threshold:
                        continue
                    
                    # 轉換座標為polygon格式
                    polygon = []
                    for point in mask:
                        x = float(point[0])
                        y = float(point[1])
                        polygon.extend([x, y])
                    
                    # 確保polygon至少有3個點
                    if len(polygon) < 6:  # 至少3個點(x,y)
                        continue
                    
                    # 計算 bounding box
                    x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
                    y_coords = [polygon[i] for i in range(1, len(polygon), 2)]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    
                    # 計算polygon面積（更精確的面積計算）
                    area = calculate_polygon_area(polygon)
                    
                    # 添加標註（標準COCO格式）
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(cls) + 1,  # COCO 格式從 1 開始
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": area,
                        "segmentation": [polygon],  # polygon座標列表
                        "iscrowd": 0,
                        "score": float(conf)  # 添加置信度分數
                    }
                    
                    coco_data["annotations"].append(annotation)
                    annotation_id += 1
        
        image_id += 1
    
    # 保存 COCO JSON 文件
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== 轉換完成 ===")
    print(f"處理圖片數量: {len(coco_data['images'])}")
    print(f"生成標註數量: {len(coco_data['annotations'])}")
    print(f"類別數量: {len(coco_data['categories'])}")
    print(f"COCO JSON 已保存到: {output_json_path}")
    
    return coco_data

def calculate_polygon_area(polygon):
    """
    使用 Shoelace 公式計算polygon面積
    
    Args:
        polygon: [x1, y1, x2, y2, ..., xn, yn] 格式的座標列表
        
    Returns:
        area: polygon面積
    """
    if len(polygon) < 6:  # 至少需要3個點
        return 0.0
    
    x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
    y_coords = [polygon[i] for i in range(1, len(polygon), 2)]
    
    n = len(x_coords)
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        area += x_coords[i] * y_coords[j]
        area -= x_coords[j] * y_coords[i]
    
    return abs(area) / 2.0

def batch_predict_single_files(model_path, images_dir, output_dir, class_names, confidence_threshold=0.5):
    """
    為每張圖片生成單獨的 COCO JSON 文件
    
    Args:
        model_path: YOLOv8 模型權重路径
        images_dir: 圖片資料夾路径
        output_dir: 輸出資料夾路径
        class_names: 類別名稱列表
        confidence_threshold: 信心度閾值
    """
    model = YOLO(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = set()  # 使用 set 避免重複
    
    for ext in image_extensions:
        image_files.update(Path(images_dir).glob(f'*{ext}'))
        image_files.update(Path(images_dir).glob(f'*{ext.upper()}'))
    
    image_files = list(image_files)
    print(f"找到 {len(image_files)} 張圖片")
    
    for image_path in image_files:
        print(f"處理圖片: {image_path}")
        
        # 讀取圖片獲取尺寸
        img = cv2.imread(str(image_path))
        if img is None:
            continue
            
        height, width = img.shape[:2]
        
        # 初始化單張圖片的標準 COCO 格式
        single_coco = {
            "info": {
                "description": f"YOLOv8 annotation for {image_path.name}",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "YOLOv8 Auto Annotation Tool",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "images": [{
                "id": 1,
                "width": width,
                "height": height,
                "file_name": image_path.name,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }],
            "annotations": [],
            "categories": []
        }
        
        # 添加類別信息
        for idx, class_name in enumerate(class_names):
            single_coco["categories"].append({
                "id": idx + 1,
                "name": class_name,
                "supercategory": "object"
            })
        
        # 預測
        results = model(str(image_path), conf=confidence_threshold)
        annotation_id = 1
        
        for result in results:
            if result.masks is not None and len(result.masks) > 0:
                masks = result.masks.xy
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for mask, cls, conf in zip(masks, classes, confidences):
                    if conf < confidence_threshold:
                        continue
                    
                    # 轉換多邊形座標
                    polygon = []
                    for point in mask:
                        polygon.extend([float(point[0]), float(point[1])])
                    
                    if len(polygon) < 6:  # 至少3個點
                        continue
                    
                    # 計算 bounding box
                    x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
                    y_coords = [polygon[i] for i in range(1, len(polygon), 2)]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    
                    # 計算面積
                    area = calculate_polygon_area(polygon)
                    
                    single_coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": 1,
                        "category_id": int(cls) + 1,
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": area,
                        "segmentation": [polygon],
                        "iscrowd": 0,
                        "score": float(conf)
                    })
                    
                    annotation_id += 1
        
        # 保存單張圖片的 JSON
        output_json = os.path.join(output_dir, f"{image_path.stem}_coco.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(single_coco, f, indent=2, ensure_ascii=False)
        
        print(f"  -> 保存到: {output_json}")

def visualize_coco_annotations(image_path, coco_json_path, output_path=None):
    """
    可視化 COCO 標註結果
    
    Args:
        image_path: 原始圖片路徑
        coco_json_path: COCO JSON 檔案路徑
        output_path: 輸出可視化圖片路徑（可選）
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPLPolygon
    import matplotlib.patches as patches
    
    # 讀取圖片
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 讀取COCO JSON
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 創建類別ID到名稱的映射
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_rgb)
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    for i, ann in enumerate(coco_data['annotations']):
        color = colors[i % len(colors)]
        
        # 繪製邊界框
        bbox = ann['bbox']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                               linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # 繪製分割polygon
        for seg in ann['segmentation']:
            poly_points = []
            for j in range(0, len(seg), 2):
                if j + 1 < len(seg):
                    poly_points.append([seg[j], seg[j + 1]])
            
            if len(poly_points) > 2:
                polygon = MPLPolygon(poly_points, fill=True, 
                                   facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
                ax.add_patch(polygon)
        
        # 添加標籤
        cat_name = cat_id_to_name.get(ann['category_id'], 'Unknown')
        score = ann.get('score', 0.0)
        ax.text(bbox[0], bbox[1] - 5, 
               f"{cat_name}: {score:.2f}",
               color=color, fontsize=10, weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_title(f'COCO Annotations: {Path(image_path).name}')
    ax.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"可視化結果保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()

def validate_coco_format(coco_json_path):
    """
    驗證 COCO 格式是否正確
    
    Args:
        coco_json_path: COCO JSON 檔案路徑
        
    Returns:
        bool: 驗證是否通過
    """
    try:
        with open(coco_json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 檢查必要的欄位
        required_fields = ['info', 'licenses', 'images', 'annotations', 'categories']
        for field in required_fields:
            if field not in coco_data:
                print(f"缺少必要欄位: {field}")
                return False
        
        print("COCO 格式驗證通過！")
        print(f"圖片數量: {len(coco_data['images'])}")
        print(f"標註數量: {len(coco_data['annotations'])}")
        print(f"類別數量: {len(coco_data['categories'])}")
        
        return True
        
    except Exception as e:
        print(f"COCO 格式驗證失敗: {str(e)}")
        return False

# 使用範例
if __name__ == "__main__":
    # 配置參數
    MODEL_PATH = "D:/呈仲/ultralytics/weights/v3.pt"  # YOLOv8 權重文件
    IMAGES_DIR = "D:/ChengChung/End_face_measurement/dataset/test/images"   # 圖片資料夾
    OUTPUT_JSON = "D:/ChengChung/End_face_measurement/dataset/test/coco_annotations.json"     # 輸出的COCO標註文件
    
    # 類別名稱（按照訓練時的順序）
    CLASS_NAMES = ["rectangular-segmentation"]  # 您的實際類別
    
    print("=== YOLOv8 分割結果轉 COCO JSON ===")
    
    # 方法1：生成單一 COCO JSON 文件（包含所有圖片）
    coco_data = yolov8_predict_to_coco(
        model_path=MODEL_PATH,
        images_dir=IMAGES_DIR,
        output_json_path=OUTPUT_JSON,
        class_names=CLASS_NAMES,
        confidence_threshold=0.3  # 可以調低一點，後續手動篩選
    )
    
    # 驗證COCO格式
    print("\n=== 驗證 COCO 格式 ===")
    validate_coco_format(OUTPUT_JSON)
    
    # 可視化第一張圖片的標註（如果有的話）
    if len(coco_data['images']) > 0:
        first_image = coco_data['images'][0]
        image_path = Path(IMAGES_DIR) / first_image['file_name']
        if image_path.exists():
            print(f"\n=== 可視化標註結果 ===")
            visualize_coco_annotations(
                image_path, 
                OUTPUT_JSON, 
                "visualization_output.jpg"
            )
    
    # 方法2：為每張圖片生成單獨的 COCO JSON 文件（取消註解使用）
    # print("\n=== 生成單獨的 JSON 文件 ===")
    # batch_predict_single_files(
    #     model_path=MODEL_PATH,
    #     images_dir=IMAGES_DIR,
    #     output_dir="output_coco_jsons",
    #     class_names=CLASS_NAMES,
    #     confidence_threshold=0.3
    # )
    
    print("\n=== 完成！ ===")
    print("您現在可以：")
    print("1. 直接使用這個COCO JSON進行模型訓練")
    print("2. 導入到CVAT、labelme等標註工具中進行編輯")
    print("3. 使用COCO API進行評估和分析")