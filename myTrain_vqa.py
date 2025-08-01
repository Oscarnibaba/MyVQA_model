import argparse
import os
import sys
import ruamel.yaml as yaml
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
    header = 'Generate VQA result:'
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

def main(args, config):
    if args.distributed:
        utils.init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    utils.set_seed(args.seed + utils.get_rank())

    print('Creating vqa {} datasets'.format(args.dataset_use))
    datasets = create_dataset(args.dataset_use, config)
    has_val = 'val_file' in config[args.dataset_use] and len(config[args.dataset_use]['val_file']) > 0

    if has_val:
        print('train:', len(datasets[0]), 'test:', len(datasets[1]), 'val:', len(datasets[2]))
        samplers = create_sampler(datasets, [True, False, False], utils.get_world_size(), utils.get_rank()) if args.distributed else [None, None, None]
        loaders = create_loader(datasets, samplers, batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']], num_workers=[4, 4, 4], is_trains=[True, False, False], collate_fns=[vqa_collate_fn, None, None])
        train_loader, test_loader, val_loader  = loaders
    else:
        print('train:', len(datasets[0]), 'test:', len(datasets[1]))
        samplers = create_sampler(datasets, [True, False], utils.get_world_size(), utils.get_rank()) if args.distributed else [None, None]
        loaders = create_loader(datasets, samplers, batch_size=[config['batch_size_train'], config['batch_size_test']], num_workers=[4, 4], is_trains=[True, False], collate_fns=[vqa_collate_fn, None])
        train_loader, test_loader = loaders
        val_loader = None

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    answer_list = test_loader.dataset.answer_list
    label2id = {a: i for i, a in enumerate(answer_list)}
    id2label = {i: a for a, i in label2id.items()}

    print("Creating classifier model")
    model = VQA_Classifier(config=config, tokenizer=tokenizer, text_encoder=args.text_encoder, num_answer_classes=len(answer_list)).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if 'visual_encoder.pos_embed' in state_dict:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)
            if 'text_encoder' in key:
                if 'layer' in key:
                    encoder_keys = key.split('.')
                    layer_num = int(encoder_keys[4])
                    if layer_num < 6:
                        del state_dict[key]
                        continue
                    else:
                        decoder_layer_num = (layer_num - 6)
                        encoder_keys[4] = str(decoder_layer_num)
                        encoder_key = '.'.join(encoder_keys)
                else:
                    encoder_key = key
                decoder_key = encoder_key.replace('text_encoder', 'text_decoder')
                state_dict[decoder_key] = state_dict[key]

                del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=False)
        print('Loaded checkpoint from', args.checkpoint)
        print("Missing keys:", msg.missing_keys)
        print("Unexpected keys:", msg.unexpected_keys)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module if hasattr(model, 'module') else model

    print("\nStart training\n")
    start_epoch = 0
    start_time = time.time()
    prefix = Path(args.checkpoint).stem if args.checkpoint else 'no_ckpt'

    for epoch in range(start_epoch, config['max_epoch']):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
        train(model, train_loader, optimizer, epoch, device, config, label2id)

        if epoch > 0:
            if val_loader is not None:
                val_result = evaluation(model, val_loader, device, config, id2label)
                all_val = gather_evaluation_results(val_result)
                if utils.is_main_process():
                    val_json = os.path.join(args.result_dir, f'{prefix}_val_result_{epoch}.json')
                    with open(val_json, 'w') as f:
                        json.dump(all_val, f)
                    evaluate_vqa_accuracy(val_json, config[args.dataset_use]['val_file'][0], args.acc_output_file, epoch)

            test_result = evaluation(model, test_loader, device, config, id2label)
            all_test = gather_evaluation_results(test_result)
            if utils.is_main_process():
                test_json = os.path.join(args.result_dir, f'{prefix}_test_result_{epoch}.json')
                with open(test_json, 'w') as f:
                    json.dump(all_test, f)
                evaluate_vqa_accuracy(test_json, config[args.dataset_use]['test_file'][0], args.acc_output_file, epoch)
                torch.save({'model': model_without_ddp.state_dict()}, os.path.join(args.output_dir, f'{prefix}_epoch_{epoch}.pth'))

        if args.distributed:
            dist.barrier()

    print('Training time', str(datetime.timedelta(seconds=int(time.time() - start_time))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_use', default='rad')
    parser.add_argument('--is_save_path', default=False)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_suffix', default='')
    parser.add_argument('--output_dir', default='/data/nfs/qiuchen/hyf/code/MyVQA_model-main/output')
    parser.add_argument('--text_encoder', default='/data/nfs/qiuchen/hyf/pretrained_ckp/bert-base-uncased')
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

    print("config:", config)
    print("args:", args)
    main(args, config)
