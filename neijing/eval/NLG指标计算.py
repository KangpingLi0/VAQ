import json
import os
import csv
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def coco_caption_eval(data):
    """
    在内存中动态构建 COCO 格式并进行评测
    """
    coco_gt = COCO()
    dataset = {
        "images": [],
        "annotations": [],
        "type": "captions",
        "info": "",
        "licenses": ""
    }
    
    predictions = []
    
    for i, item in enumerate(data):
        image_id = i  
        
        dataset["images"].append({"id": image_id})
        
        label = item.get("label", "")
        if isinstance(label, list):
            for j, l in enumerate(label):
                dataset["annotations"].append({
                    "image_id": image_id,
                    "id": int(f"{image_id}00{j}"),
                    "caption": str(l),
                    "label": str(l)
                })
        else:
            dataset["annotations"].append({
                "image_id": image_id,
                "id": image_id,
                "caption": str(label),
                "label": str(label)
            })
            
        pred = item.get("pred", "")
        if isinstance(pred, list):
            pred = " ".join(pred)
            
        predictions.append({
            "image_id": image_id,
            "caption": str(pred),
            "pred": str(pred)  
        })

    coco_gt.dataset = dataset
    coco_gt.createIndex()
    
    coco_res = coco_gt.loadRes(predictions)
    
    coco_eval = COCOEvalCap(coco_gt, coco_res)
    coco_eval.params["image_id"] = coco_res.getImgIds()
    coco_eval.evaluate()
    
    return coco_eval.eval

def main():
    # 1. 在这里直接定义你要评测的 JSON 文件列表
    file_list = [
        "/work/data/lxr/code/结果分析/随机测试/J1.json",
        "/work/data/lxr/code/结果分析/随机测试/J2.json",
    ]
    
    # 2. 定义输出的 CSV 文件路径
    output_csv = "results_eval/_NLG_metrics.csv"

    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_results = []
    metrics_keys = set()

    # 遍历处理每一个文件
    for file_path in file_list:
        # 增加一个容错判断，文件不存在直接跳过
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping -> {file_path}")
            continue
            
        print(f"Processing: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 计算指标
            eval_metrics = coco_caption_eval(data)
            
            # 记录当前文件的结果
            row_data = {"filename": os.path.basename(file_path)}
            for metric, score in eval_metrics.items():
                row_data[metric] = f"{score:.6f}"
                metrics_keys.add(metric)  # 收集所有的指标名称用于生成 CSV 表头
                
            all_results.append(row_data)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # 如果没有任何成功处理的文件，直接退出
    if not all_results:
        print("No valid results to save. Exiting.")
        return

    # 整理 CSV 表头（filename 在第一列，其余指标按字母排序）
    sorted_metrics = sorted(list(metrics_keys))
    fieldnames = ["filename"] + sorted_metrics

    # 写入 CSV 文件
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
            
    print(f"\nSuccessfully saved all results to: {output_csv}")

if __name__ == "__main__":
    main()