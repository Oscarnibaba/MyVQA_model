import argparse
import os
import sys
import ruamel_yaml as yaml
import time
import datetime
import json
from pathlib import Path
import torch
import torch.distributed as dist
import logging
from models.Mymodel_vqa import VQA_Classifier
from models.vision.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn
from utils import cosine_lr_schedule
from EvalAcc_vqa import evaluate_vqa_accuracy, gather_evaluation_results

def train(model, data_loader, optimizer, epoch, device, config, label2id):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, question, answer) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        label = torch.tensor([label2id[a] for a in answer], device=device)

        loss = model(image, question, labels=label, train=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, device, config, id2label):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    result = []

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        preds, probs = model(image, question, train=False)

        for i, ques_id in enumerate(question_id):
            ques_id = int(ques_id.item())
            pred_answer = id2label[preds[i].item()]
            result.append({"qid": ques_id, "answer": pred_answer})
    return result

def main(args, config, acc_output_file):
    if args.distributed:
        utils.init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    utils.set_seed(args.seed + utils.get_rank())

    print('Creating vqa {} datasets'.format(args.dataset_use))
    datasets = create_dataset(args.dataset_use, config)
    print('train dataset size: ', len(datasets[0]))
    print('test dataset size: ', len(datasets[1]))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    answer_list = datasets[1].answer_list  # 更保险地使用测试集的 answer list
    label2id = {a: i for i, a in enumerate(answer_list)}
    id2label = {i: a for a, i in label2id.items()}

    print("Creating classifier model")
    model = VQA_Classifier(config=config, tokenizer=tokenizer, text_encoder=args.text_encoder, num_answer_classes=len(answer_list))
    model = model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        if 'visual_encoder.pos_embed' in state_dict:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        msg = model.load_state_dict(state_dict, strict=False)
        print('Loaded checkpoint from %s' % args.checkpoint)
        print("Missing keys:", msg.missing_keys)
        print("Unexpected keys:", msg.unexpected_keys)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module if hasattr(model, 'module') else model

    start_epoch = 0
    print("\nStart training\n")
    start_time = time.time()

    for epoch in range(start_epoch, config['max_epoch']):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
        train(model, train_loader, optimizer, epoch, device, config, label2id)

        prefix = Path(args.checkpoint).stem if args.checkpoint else 'no_ckpt'

        if epoch > 9:
            vqa_result = evaluation(model, test_loader, device, config, id2label)
            all_results = gather_evaluation_results(vqa_result)

        if utils.is_main_process() and epoch > 9:
            json_file_path = os.path.join(args.result_dir, f'{prefix}_vqa_result_{epoch}.json')
            with open(json_file_path, 'w') as f:
                json.dump(all_results, f)
            res_file_path = '%s/result/%s_vqa_result_%d.json' % (args.output_dir, prefix, epoch)
            evaluate_vqa_accuracy(res_file_path, config[args.dataset_use]['test_file'][0], args.acc_output_file, epoch)

            # 保存模型
            save_obj = {'model': model_without_ddp.state_dict()}
            torch.save(save_obj, os.path.join(args.output_dir, f'{prefix}_epoch_{epoch}.pth'))

        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_use', default='rad')
    parser.add_argument('--is_save_path', default=False)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_suffix', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--distributed', default=False, type=bool)

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.dataset_use, args.output_suffix)
    config = yaml.load(open('./configs/VQA.yaml', 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    sys.stdout = utils.Logger(filename=os.path.join(args.output_dir, "log.txt"), stream=sys.stdout)
    args.acc_output_file = os.path.join(args.result_dir, 'accuracy_log.json')

    print("config: ", config)
    print("args: ", args)
    main(args, config)
