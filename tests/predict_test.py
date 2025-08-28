from PIL import Image, ImageDraw
from pathlib import Path
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# # Run inference on 'bus.jpg'
# results = model(["D:/yolov5-master/data/Falldown/images\6cctv_100754_101043_27.jpg", "https://ultralytics.com/images/zidane.jpg"])  # results list

# # Visualize the results
# for i, r in enumerate(results):
#     # Plot results image
#     im_bgr = r.plot()  # BGR-order numpy array
#     im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

#     # Show results to screen (in supported environments)
#     r.show()

#     # Save results to disk
#     r.save(filename=f"results{i}.jpg")

folder = Path('D:/ultralytics/test_data/')
save_path = r'D:/ultralytics/result/'
data_list = [p.as_posix() for p in folder.rglob('*.jpg')]
# print(data_list[0])
# results = model(data_list[0])
# results = model(data_list[0])  # results list
# for i, r in enumerate(results):
#     im_bgr = r.plot()
#     im_rgb = Image.fromarray(im_bgr[..., ::-1])
#     r.save(filename=f"{save_path}result.jpg")
for i in data_list[0:1]:
    name = i.split("/")[-1].split(".")[0]
    result = model(i)
    for r in result:
        boxes = r.boxes
        # masks = r.masks
        # keypoints = r.keypoints
        # probs = r.probs
        # obb = r.obb
        # r.show()
        # r.save(f"{save_path}{name}.jpg")
        # print(r)


img = Image.open(data_list[0])  # 請換成你的圖片檔案路徑

# 建立可以畫圖的物件
draw = ImageDraw.Draw(img)

# 指定座標
x, y = 1.6942e+03, 8.3382e+02

# 畫一個小紅點（可以用畫圓來模擬點）
radius = 5
draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')

# 顯示圖片
img.show()