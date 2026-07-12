import json

# 从文件加载关键词数据
with open('eval/部位描述关键词.json', 'r', encoding='utf-8') as f:
    keyword_description = json.load(f)

with open('eval/诊断结论关键词_hzh.json', 'r', encoding='utf-8') as f:
    keyword_conclusion = json.load(f)

def parse_medical_report(report):
    """解析医学报告，提取各部位的描述和诊断结论"""
    report_lines = report.split('\n')
    parsed_info = {}

    for line in report_lines:
        if line.startswith('- '):
            part = line[2:].split('：')[0]
            # try:
            description = line[2:].split('：')[1]
            # except:
            #     continue
            parsed_info[part] = description
        if '诊断结论' in line:
            diagnosis = line.split('：')[1]
            parsed_info['诊断结论'] = diagnosis

    return parsed_info

def extract_keywords_from_description(parsed_report, keyword_data):
    """根据关键词数据从报告的部位描述中提取相关信息"""
    extracted_keywords = []

    for part, description in parsed_report.items():

        description_parts = description.split('，')

        for part_class, part_keywords in keyword_data.items():
            if part not in part_class:
                continue
            
            for keyword_class, keywords in part_keywords.items():
                for keyword, synonym_info in keywords.items():
                    all_synonyms = synonym_info['同义词'] + [keyword]
                    for synonym in all_synonyms:
                        for i, desc in enumerate(description_parts):
                            if '未见' in desc or '无' in desc:
                                continue
                            if '配合不出现' in synonym_info and synonym_info['配合不出现'] in desc:
                                continue
                            if '配合使用' in synonym_info and synonym_info['配合使用'] and synonym_info['配合使用'] not in description:
                                continue
                            if synonym in desc:
                                description_parts[i] = desc.replace(synonym, '')
                                if '含义' in synonym_info and synonym_info['含义']:
                                    extracted_keywords.append(f"{part}-{synonym_info['含义']}")
                                else:
                                    extracted_keywords.append(f"{part}-{keyword}")

    return extracted_keywords

def extract_keywords_from_conclusion(parsed_report, keyword_data):
    """根据关键词数据从报告的诊断结论中提取相关信息"""
    extracted_keywords = []

    if '诊断结论' not in parsed_report:
        return extracted_keywords

    conclusion = parsed_report['诊断结论']
    conclusion_parts = conclusion.split('；')

    for part_or_disease, part_keywords in keyword_data.items():
        if part_or_disease == "子部位":
            continue
        for keyword_class, keywords in part_keywords.items():
            for keyword, synonym_info in keywords.items():
                all_synonyms = synonym_info['同义词'] + [keyword]
                for synonym in all_synonyms:
                    for i, conclusion in enumerate(conclusion_parts):
                        if '未见' in conclusion or '无' in conclusion:
                            continue
                        if '配合不出现' in synonym_info and synonym_info['配合不出现'] in conclusion:
                            continue
                        if '配合使用' in synonym_info and synonym_info['配合使用'] and synonym_info['配合使用'] not in conclusion:
                            continue
                        if synonym in conclusion:
                            conclusion_parts[i] = conclusion.replace(synonym, '')
                            if '含义' in synonym_info and synonym_info['含义']:
                                extracted_keywords.append(f"诊断结论-{synonym_info['含义']}")
                            else:
                                extracted_keywords.append(f"诊断结论-{keyword}")
                                
    return extracted_keywords

import json
import glob
# 假设 `parse_medical_report` 和 `extract_keywords_from_description` 已经定义

# 加载数据
for json_file in glob.glob('/work/data/public/gastrohun_ours_model.json'):
    with open(json_file, 'r', encoding='utf-8') as f:
        test_results = json.load(f)

    # 用于存储计算出的精确度、召回率和F1分数
    all_precision = []
    all_recall = []
    all_f1 = []
    
    # 逐个处理每个case
    for case in test_results:
        pred = case['predict']
        # print(pred)
        label = case['answer']

        # 提取pred和label的关键词
        parse_pred = parse_medical_report(pred)
        # extracted_info_pred =  \
        #     extract_keywords_from_conclusion(parse_pred, keyword_conclusion) \
        #     + extract_keywords_from_description(parse_pred, keyword_description)
        
        # extracted_info_pred =  extract_keywords_from_description(parse_pred, keyword_description)
        extracted_info_pred =  extract_keywords_from_conclusion(parse_pred, keyword_conclusion)
        
        # print(extracted_info_pred)
        parse_label = parse_medical_report(label)
        # extracted_info_label = \
        #     extract_keywords_from_conclusion(parse_label, keyword_conclusion) \
        #     + extract_keywords_from_description(parse_label, keyword_description)
        
        # extracted_info_label = extract_keywords_from_description(parse_label, keyword_description)
        extracted_info_label = extract_keywords_from_conclusion(parse_label, keyword_conclusion)
        # 转换为集合，方便计算交集
        set_pred = set(extracted_info_pred)
        set_label = set(extracted_info_label)
        # print(extracted_info_label)
        # if case['image_id'] == "hfyy_202301_00027":
        #     print("pred",pred)
        #     print("label",label)
        #     print("extracted_info_pred",extracted_info_pred)
        #     print("extracted_info_label",extracted_info_label)

        # 计算 Precision, Recall 和 F1 分数
        if not set_pred and not set_label:
            precision = recall = f1 = 1.0  # 如果两者都为空，定义为完美匹配
        else:
            # True Positives (TP): 提取出的关键词在label中存在
            tp = len(set_pred.intersection(set_label))
            # False Positives (FP): 提取出的关键词不在label中
            fp = len(set_pred - set_label)
            # False Negatives (FN): label中存在的关键词未被提取出来
            fn = len(set_label - set_pred)

            # 计算精确度 (Precision)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # 计算召回率 (Recall)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # 计算F1分数
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 存储结果
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        
        # 打印输出
        # order = ['食道', '贲门', '胃底', '胃体', '胃角', '胃窦', '幽门', '十二指肠球部', '十二指肠降部', '诊断结论']
        # print(f"ID: {case['image_id']}")
        # print(f"Pred: {pred} \n")
        # print(f"Label: {label} \n")
        # print(f"Pred Extracted Info: {sorted(list(set_pred), key=lambda x: order.index(x.split('-')[0]))}")
        # print(f"Label Extracted Info: {sorted(list(set_label), key=lambda x: order.index(x.split('-')[0]))}")
        # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        # print("========================================")
        # break

    # 计算总体的平均精确度、召回率和F1分数
    avg_precision = sum(all_precision) / len(all_precision)
    avg_recall = sum(all_recall) / len(all_recall)
    avg_f1 = sum(all_f1) / len(all_f1)

    print(f"Results for file: {json_file}, data length: {len(test_results)}" )
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
