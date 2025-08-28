import os
import cv2
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from pathlib import Path
import argparse

def create_voc_annotation(image_path, detections, output_dir, class_names):
    """
    創建VOC格式的XML標註檔案
    
    Args:
        image_path: 影像檔案路徑
        detections: YOLO預測結果
        output_dir: 輸出目錄
        class_names: 類別名稱列表
    """
    # 讀取影像獲取尺寸
    image = cv2.imread(str(image_path))
    height, width, channels = image.shape
    
    # 創建XML根節點
    annotation = ET.Element('annotation')
    
    # 基本資訊
    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'images'
    
    filename = ET.SubElement(annotation, 'filename')
    filename.text = os.path.basename(image_path)
    
    path = ET.SubElement(annotation, 'path')
    path.text = str(image_path)
    
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
    
    size = ET.SubElement(annotation, 'size')
    size_width = ET.SubElement(size, 'width')
    size_width.text = str(width)
    size_height = ET.SubElement(size, 'height')
    size_height.text = str(height)
    size_depth = ET.SubElement(size, 'depth')
    size_depth.text = str(channels)
    
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'
    
    # 處理每個偵測結果
    for detection in detections:
        # 獲取邊界框座標 (xyxy format)
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        confidence = detection.conf[0].cpu().numpy()
        class_id = int(detection.cls[0].cpu().numpy())
        
        # 確保座標在影像範圍內
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width - 1, int(x2)))
        y2 = max(0, min(height - 1, int(y2)))
        
        # 創建物件節點
        obj = ET.SubElement(annotation, 'object')
        
        name = ET.SubElement(obj, 'name')
        name.text = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
        
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '0'
        
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = '0'
        
        confidence_elem = ET.SubElement(obj, 'confidence')
        confidence_elem.text = f'{confidence:.6f}'
        
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(x1)
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(y1)
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(x2)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(y2)
    
    # 儲存XML檔案
    output_file = os.path.join(output_dir, f"{Path(image_path).stem}.xml")
    tree = ET.ElementTree(annotation)
    
    # 格式化XML
    ET.indent(tree, space="  ", level=0)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    return output_file

def get_image_files(images_path):
    """
    獲取影像檔案列表，避免重複
    
    Args:
        images_path: 影像目錄或單一影像檔案路徑
    
    Returns:
        list: 唯一的影像檔案路徑列表
    """
    if os.path.isfile(images_path):
        return [images_path]
    
    # 支援的影像格式（不區分大小寫）
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_files = set()  # 使用set避免重複
    
    # 遍歷目錄中的所有檔案
    for file_path in Path(images_path).iterdir():
        if file_path.is_file():
            # 檢查副檔名（轉為小寫比較）
            if file_path.suffix.lower() in extensions:
                image_files.add(str(file_path))
    
    return sorted(list(image_files))  # 轉為排序的列表

def predict_and_convert_to_voc(model_path, images_path, output_dir, confidence_threshold=0.25):
    """
    使用YOLO模型預測影像並轉換為VOC格式
    
    Args:
        model_path: YOLO .pt模型檔案路徑
        images_path: 影像目錄或單一影像檔案路徑
        output_dir: VOC XML檔案輸出目錄
        confidence_threshold: 信心度閾值
    """
    # 載入YOLO模型
    print(f"載入YOLO模型: {model_path}")
    model = YOLO(model_path)
    
    # 獲取類別名稱
    class_names = model.names
    print(f"類別數量: {len(class_names)}")
    print(f"類別名稱: {list(class_names.values())}")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 獲取影像檔案列表
    image_files = get_image_files(images_path)
    
    if not image_files:
        print(f"在 {images_path} 中沒有找到影像檔案")
        return
    
    print(f"找到 {len(image_files)} 個影像檔案")
    
    # 列出找到的檔案
    print("檔案列表:")
    for i, file_path in enumerate(image_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")
    
    processed_count = 0
    
    # 處理每個影像
    for image_file in image_files:
        print(f"\n處理: {os.path.basename(image_file)}")
        
        try:
            # 進行預測
            results = model(image_file, conf=confidence_threshold, verbose=False)
            
            # 獲取偵測結果
            detections = results[0].boxes
            
            if detections is not None and len(detections) > 0:
                print(f"  偵測到 {len(detections)} 個物件")
                
                # 創建VOC標註檔案
                xml_file = create_voc_annotation(image_file, detections, output_dir, class_names)
                print(f"  已儲存: {os.path.basename(xml_file)}")
                processed_count += 1
            else:
                print(f"  未偵測到任何物件 (信心度 >= {confidence_threshold})")
                # 即使沒有偵測結果也創建空的XML檔案
                xml_file = create_voc_annotation(image_file, [], output_dir, class_names)
                print(f"  已儲存空標註: {os.path.basename(xml_file)}")
                processed_count += 1
                
        except Exception as e:
            print(f"  錯誤: {str(e)}")
    
    print(f"\n完成! 共處理 {processed_count} 個檔案")
    print(f"VOC XML檔案已儲存至: {output_dir}")

'''
default => 可以更改屬於自己的pt檔、未標註影像path、輸出VOC格式的資料夾位置
'''
def main():
    parser = argparse.ArgumentParser(description='使用YOLO11預測影像並輸出VOC格式標註檔案')
    parser.add_argument('--model', '-m', default='D:/呈仲/ultralytics/weights/best_0826.pt', help='YOLO .pt模型檔案路徑')
    parser.add_argument('--images', '-i', default='D:/呈仲/ultralytics/unlabeled_images/image', help='影像目錄或單一影像檔案路徑')
    parser.add_argument('--output', '-o', default='D:/呈仲/ultralytics/labels', help='VOC XML檔案輸出目錄')
    # parser.add_argument('--conf', '-c', type=float, default=0.25, help='信心度閾值 (預設: 0.25)')
    
    args = parser.parse_args()
    
    # 檢查模型檔案是否存在
    if not os.path.exists(args.model):
        print(f"錯誤: 找不到模型檔案 {args.model}")
        return
    
    # 檢查影像路徑是否存在
    if not os.path.exists(args.images):
        print(f"錯誤: 找不到影像路徑 {args.images}")
        return
    
    predict_and_convert_to_voc(args.model, args.images, args.output)

if __name__ == "__main__":
    # 如果直接執行腳本，使用命令列參數
    main()