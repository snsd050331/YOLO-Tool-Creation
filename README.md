# YOLO系列-自製資料集準備
## 1st - Create folder
1.創建data資料夾，data下在創建4個資料夾Images、Labels、Annotations、ImageSets  
2.將自己的資料集放入Images、Labels中(影像跟標註分開)  
3.在data資料夾下創建Annotations、ImageSets
## 2nd - makeTxt.py
程式會在data/ImageSets/下生成  
trainval.txt  
train.txt  
val.txt  
test.txt
## 3rd - voc_label.py
更改[classes]類別，將自己預訊練的類別寫上  
運行過程中會顯示那些資料已經轉成YOLO txt格式
## data_spilt.py
是將資料拆分比例數量[train/validation/test] => 0.7 : 0.2 : 0.1 => 可以[split_data]參數中更改  
source_folder是原始路徑  
output_folder是輸出路徑  
*主要是YOLO系列中CLS分類資料集製造需要使用的程式
## cuttoxml.py - 輸出PASCAL VOC格式
將訓練好的權重拿來快速預測未標註影像 => 加速標註效率、查看模型標註效果
python cuttoxml.py --model XXX.pt --images XXX --output XXX  
1.images => 影像目錄或單一影像檔案路徑  
2.output => VOC XML檔案輸出目錄  
3.或可以在程式中更改default位置，直接運行
