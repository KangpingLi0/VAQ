from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
import json
import re
def coco_caption_eval(results_file, annotation_file):
    coco = COCO(annotation_file)
    with open(results_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if raw and "caption" in raw[0]:
        coco_results = raw
    else:
        coco_results = [
            {"image_id": x["image_id"], "caption": x["pred"], "pred": x["pred"]}
            for x in raw
        ]
    coco_result = coco.loadRes(coco_results)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()
    return coco_eval
# gt文件参考这个：/work/model/xieqiang/推理和评测/ascend_llama_factory_huaxi_union_2022_05_to_2023_test_14694cases_gt.json

if __name__ == "__main__":
    RESULTS_FILE = "/work/model/xieqiang/推理和评测/union_2018_2019_2020_2021_train_sampled_review_42506cases_9epoch/results_ascend_llama_factory_huaxi_union_2022_05_to_2023_test_14694cases-1660.json"
    ANNOTATION_FILE = "/work/model/xieqiang/推理和评测/ascend_llama_factory_huaxi_union_2022_05_to_2023_test_14694cases_gt.json"
    coco_eval = coco_caption_eval(RESULTS_FILE, ANNOTATION_FILE)
    print(coco_eval.eval)
    with open("eval_scores.json", "w", encoding="utf-8") as f:
        json.dump(coco_eval.eval, f, ensure_ascii=False, indent=2)
