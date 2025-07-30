import logging
import torch.distributed as dist
import json
import os
import utils
def evaluate_vqa_accuracy(prediction_file, test_file, output_file=None, epoch=0):
    """
    加载 JSON 文件并根据问题类型计算每种类型的准确率、平均准确率（Average Accuracy）、总准确率，并将结果保存到文件（如果指定了文件路径）。

    参数:
    prediction_file (str): 包含预测答案的 JSON 文件路径
    test_file (str): 包含测试数据的 JSON 文件路径
    output_file (str, optional): 保存结果的 JSON 文件路径

    返回:
    accuracy_by_type (dict of {str: float}): 每种问题类型的准确率
    total_accuracy (float): 总准确率
    average_accuracy (float): 平均准确率
    """
    # 加载 JSON 文件
    with open(prediction_file, 'r') as pred_file:
        predictions = json.load(pred_file)
    with open(test_file, 'r') as test_file:
        test_data = json.load(test_file)

    # 创建字典存储每种类型的统计
    accuracy_by_type = {}
    type_count = {}
    total_correct = 0
    total_count = 0

    # 将 test_data 转换为一个字典以方便查找
    test_dict = {item['qid']: item for item in test_data}

    # 遍历所有预测结果
    for pred in predictions:
        qid = pred['qid']
        predicted = pred['answer'].strip().lower()

        if qid in test_dict:
            test_info = test_dict[qid]
            ground_truth = test_info['answer'].strip().lower()
            question_type = test_info['question_type']

            if question_type not in accuracy_by_type:
                accuracy_by_type[question_type] = 0
                type_count[question_type] = 0

            if predicted == ground_truth:
                accuracy_by_type[question_type] += 1
                total_correct += 1

            type_count[question_type] += 1
            total_count += 1

    # 计算每种问题类型的准确率
    for q_type in accuracy_by_type:
        accuracy_by_type[q_type] /= type_count[q_type]

    # 计算总准确率
    total_accuracy = total_correct / total_count if total_count > 0 else 0

    # 计算平均准确率（所有类别准确率的简单平均）
    if accuracy_by_type:
        average_accuracy = sum(accuracy_by_type.values()) / len(accuracy_by_type)
    else:
        average_accuracy = 0

    print(f"Epoch: {epoch + 1}")
    for q_type, accuracy in accuracy_by_type.items():
        print(f"Question type: {q_type}: {accuracy * 100:.2f}")
    print(f"Total Accuracy: {total_accuracy * 100:.2f}")
    print(f"Average Accuracy: {average_accuracy * 100:.2f}")

    # 如果指定了输出文件路径，将结果保存到文件
    if output_file:
        if os.path.exists(output_file):
            with open(output_file, 'r') as outfile:
                results = json.load(outfile)
        else:
            results = []

        results.append({
            "epoch": epoch + 1,
            "accuracy_by_type": {key: round(value * 100, 2) for key, value in accuracy_by_type.items()},
            "total_accuracy": round(total_accuracy * 100, 2),
            "average_accuracy": round(average_accuracy * 100, 2)
        })

        with open(output_file, 'w') as outfile:
            json.dump(results, outfile, indent=4)

    return accuracy_by_type, total_accuracy, average_accuracy

def gather_evaluation_results(vqa_result):
    """
    使用分布式收集评估结果，并通过唯一的 qid 去重。
    """
    logging.info(f"Process {dist.get_rank()} - vqa_result: {vqa_result}")

    all_results = {} if utils.is_main_process() else None

    gathered_results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_results, vqa_result)

    logging.info(f"Process {dist.get_rank()} - gathered_results: {gathered_results}")

    if utils.is_main_process():
        for result in gathered_results:
            if result is not None:
                for item in result:
                    qid = item['qid']
                    if qid not in all_results:
                        all_results[qid] = item

    return list(all_results.values()) if all_results is not None else []
# # 示例调用
# evaluate_vqa_accuracy(
#     prediction_file='/data/nfs/qiuchen/hyf/code/MUMC-main/weights/HRVQA/2025-04-27_13-16_17w/result/rs_pretrain_39_vqa_result_4.json',
#     test_file='/data/nfs/qiuchen/hyf/datasets/HRVQA/val_hrvqa.json',
#     output_file='/data/nfs/qiuchen/hyf/code/MUMC-main/weights/HRVQA/output_results.json'
# )
