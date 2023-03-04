import argparse
import json
import itertools
from transformers.utils.dummy_pt_objects import Trainer
from tools import *
from DataProcess import *
import mindspore
from mindspore import nn
from mindspore.dataset import GeneratorDataset
import CQAELModel
import random
import numpy as np
from tqdm import tqdm, trange
import tools
import math

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
mindspore.set_context(device_target='GPU', device_id=0)

class SeedDataset():
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data["all_ctxt_cross_ids"])

    def __getitem__(self, idx):
        idx = int(idx)
        # all_ctxt_cross_ids = mindspore.Tensor(self.data["all_ctxt_cross_ids"][idx]).long()
        # all_ctxt_cross_mask = mindspore.Tensor(self.data["all_ctxt_cross_mask"][idx]).long()
        # all_ctxt_cross_seg = mindspore.Tensor(self.data["all_ctxt_cross_seg"][idx]).long()
        # all_cqa_cross_ids = mindspore.Tensor(self.data["all_cqa_cross_ids"][idx]).long()
        # all_cqa_cross_mask = mindspore.Tensor(self.data["all_cqa_cross_mask"][idx]).long()
        # all_cqa_cross_seg = mindspore.Tensor(self.data["all_cqa_cross_seg"][idx]).long()
        # all_topic_cross_ids = mindspore.Tensor(self.data["all_topic_cross_ids"][idx]).long()
        # all_topic_cross_mask = mindspore.Tensor(self.data["all_topic_cross_mask"][idx]).long()
        # all_topic_cross_seg = mindspore.Tensor(self.data["all_topic_cross_seg"][idx]).long()
        # all_user_cross_ids = mindspore.Tensor(self.data["all_user_cross_ids"][idx]).long()
        # all_user_cross_mask = mindspore.Tensor(self.data["all_user_cross_mask"][idx]).long()
        # all_user_cross_seg = mindspore.Tensor(self.data["all_user_cross_seg"][idx]).long()
        # all_candidate_priors = mindspore.Tensor(self.data["all_candidate_priors"][idx]).long()
        # all_entity_masks = mindspore.Tensor(self.data["all_entity_masks"][idx]).long()
        # all_label = mindspore.Tensor(self.data["all_label"][idx]).long()

        return self.data["all_ctxt_cross_ids"][idx], \
               self.data["all_ctxt_cross_mask"][idx],\
               self.data["all_ctxt_cross_seg"][idx],\
               self.data["all_cqa_cross_ids"][idx],\
               self.data["all_cqa_cross_mask"][idx],\
               self.data["all_cqa_cross_seg"][idx],\
               self.data["all_topic_cross_ids"][idx],\
               self.data["all_topic_cross_mask"][idx],\
               self.data["all_topic_cross_seg"][idx],\
               self.data["all_user_cross_ids"][idx],\
               self.data["all_user_cross_mask"][idx],\
               self.data["all_user_cross_seg"][idx],\
               self.data["all_candidate_priors"][idx],\
               self.data["all_entity_masks"][idx],\
               self.data["all_label"][idx]

        # return self.data["all_ctxt_cross_ids"][idx], self.data["all_ctxt_cross_mask"][idx], self.data["all_ctxt_cross_seg"][idx], self.data["all_cqa_cross_ids"][idx], self.data["all_cqa_cross_mask"][idx], self.data["all_cqa_cross_seg"][idx], self.data["all_topic_cross_ids"][idx], self.data["all_topic_cross_mask"][idx], self.data["all_topic_cross_seg"][idx], self.data["all_user_cross_ids"][idx], self.data["all_user_cross_mask"][idx], self.data["all_user_cross_seg"][idx], self.data["all_candidate_priors"][idx], self.data["all_entity_masks"][idx], self.data["all_label"][idx]

def getalldata(datafile):
    f = open(datafile, 'r', encoding='utf-8')
    qa = []
    qa_index = []
    qa_json = json.load(f)
    # 遍历每一个question
    for q_index, q in enumerate(qa_json["questions"]):
        qa.append(q)
        qa_index.append(q_index)
    return list(itertools.zip_longest(qa, qa_index))


def train(model, data, optimizer, criterion, epoch_idx, logger, args, LEN_TRAIN_SET):

    model.set_train(True)
    tr_total_loss = 0
    train_accuarcy = 0
    total_train_examples = 0

    batch_count = math.ceil((data["all_ctxt_cross_ids"].shape[0]) / args.batch_size)
    logger.info('batch_count: {}'.format(batch_count))

    batch_train_targets = []
    all_ctxt_cross_ids = []
    all_ctxt_cross_mask = []
    all_ctxt_cross_seg = []
    all_cqa_cross_ids = []
    all_cqa_cross_mask = []
    all_cqa_cross_seg = []
    all_topic_cross_ids = []
    all_topic_cross_mask = []
    all_topic_cross_seg = []
    all_user_cross_ids = []
    all_user_cross_mask = []
    all_user_cross_seg = []
    all_candidate_priors = []
    all_entity_masks = []
    for i in range(batch_count):
        all_ctxt_cross_ids.append(data["all_ctxt_cross_ids"][i * args.batch_size: (i + 1) * args.batch_size])
        all_ctxt_cross_mask.append(data["all_ctxt_cross_mask"][i * args.batch_size: (i + 1) * args.batch_size])
        all_ctxt_cross_seg.append(data["all_ctxt_cross_seg"][i * args.batch_size: (i + 1) * args.batch_size])
        all_cqa_cross_ids.append(data["all_cqa_cross_ids"][i * args.batch_size: (i + 1) * args.batch_size])
        all_cqa_cross_mask.append(data["all_cqa_cross_mask"][i * args.batch_size: (i + 1) * args.batch_size])
        all_cqa_cross_seg.append(data["all_cqa_cross_seg"][i * args.batch_size: (i + 1) * args.batch_size])
        all_topic_cross_ids.append(data["all_topic_cross_ids"][i * args.batch_size: (i + 1) * args.batch_size])
        all_topic_cross_mask.append(data["all_topic_cross_mask"][i * args.batch_size: (i + 1) * args.batch_size])
        all_topic_cross_seg.append(data["all_topic_cross_seg"][i * args.batch_size: (i + 1) * args.batch_size])
        all_user_cross_ids.append(data["all_user_cross_ids"][i * args.batch_size: (i + 1) * args.batch_size])
        all_user_cross_mask.append(data["all_user_cross_mask"][i * args.batch_size: (i + 1) * args.batch_size])
        all_user_cross_seg.append(data["all_user_cross_seg"][i * args.batch_size: (i + 1) * args.batch_size])
        all_candidate_priors.append(data["all_candidate_priors"][i * args.batch_size: (i + 1) * args.batch_size])
        all_entity_masks.append(data["all_entity_masks"][i * args.batch_size: (i + 1) * args.batch_size])
        batch_train_targets.append(data["all_label"][i * args.batch_size: (i + 1) * args.batch_size])

    for i in range(batch_count):

        def forward_fn():
            logits = model(
                all_ctxt_cross_ids[i],
                all_ctxt_cross_mask[i],
                all_ctxt_cross_seg[i],
                all_cqa_cross_ids[i],
                all_cqa_cross_mask[i],
                all_cqa_cross_seg[i],
                all_topic_cross_ids[i],
                all_topic_cross_mask[i],
                all_topic_cross_seg[i],
                all_user_cross_ids[i],
                all_user_cross_mask[i],
                all_user_cross_seg[i],
                all_candidate_priors[i],
                batch_train_targets[i],
                all_entity_masks[i],
            )

            labels = mindspore.Tensor(batch_train_targets[i], dtype=mindspore.int32)
            # cast = mindspore.ops.Cast()
            # logits = cast(logits, mindspore.float32)
            loss = criterion(logits, labels)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            return loss, logits, labels

        grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        (loss, logits, labels), grads = grad_fn()
        loss = mindspore.ops.depend(loss, optimizer(grads))

        if (i + 1) % (
                args.print_tr_loss_opt_steps_interval
                * args.gradient_accumulation_steps
        ) == 0:
            logger.info(
                'Epoch [{:d}/{:d}], Iter[{:d}/{:d}], batch_train_loss:{:.9f}' \
                    .format(epoch_idx + 1, args.epochs, i+1, batch_count,
                            float(loss.item(0)))
            )
        # logger.info(
        #     'Epoch [{:d}/{:d}], Iter[{:d}/{:d}], batch_train_loss:{:.9f}' \
        #         .format(epoch_idx + 1, args.epochs, i+1, batch_count,
        #                 float(loss.item(0)))
        # )
        tr_total_loss += float(loss.item(0))
        temp_train_accuarcy = tools.accuracy(logits, labels)

        train_accuarcy += temp_train_accuarcy

    total_train_examples = batch_count * args.batch_size
    normalized_train_accuarcy = train_accuarcy/total_train_examples

    return tr_total_loss/total_train_examples, normalized_train_accuarcy


def eval(model, data, args, LEN_EVAL_SET):
    model.set_train(False)

    eval_accuarcy = 0

    batch_count = math.ceil((data["all_ctxt_cross_ids"].shape[0]) / args.batch_size)

    batch_train_targets = []
    all_ctxt_cross_ids = []
    all_ctxt_cross_mask = []
    all_ctxt_cross_seg = []
    all_cqa_cross_ids = []
    all_cqa_cross_mask = []
    all_cqa_cross_seg = []
    all_topic_cross_ids = []
    all_topic_cross_mask = []
    all_topic_cross_seg = []
    all_user_cross_ids = []
    all_user_cross_mask = []
    all_user_cross_seg = []
    all_candidate_priors = []
    all_entity_masks = []
    for i in range(batch_count):
        all_ctxt_cross_ids.append(data["all_ctxt_cross_ids"][i * args.batch_size: (i + 1) * args.batch_size])
        all_ctxt_cross_mask.append(data["all_ctxt_cross_mask"][i * args.batch_size: (i + 1) * args.batch_size])
        all_ctxt_cross_seg.append(data["all_ctxt_cross_seg"][i * args.batch_size: (i + 1) * args.batch_size])
        all_cqa_cross_ids.append(data["all_cqa_cross_ids"][i * args.batch_size: (i + 1) * args.batch_size])
        all_cqa_cross_mask.append(data["all_cqa_cross_mask"][i * args.batch_size: (i + 1) * args.batch_size])
        all_cqa_cross_seg.append(data["all_cqa_cross_seg"][i * args.batch_size: (i + 1) * args.batch_size])
        all_topic_cross_ids.append(data["all_topic_cross_ids"][i * args.batch_size: (i + 1) * args.batch_size])
        all_topic_cross_mask.append(data["all_topic_cross_mask"][i * args.batch_size: (i + 1) * args.batch_size])
        all_topic_cross_seg.append(data["all_topic_cross_seg"][i * args.batch_size: (i + 1) * args.batch_size])
        all_user_cross_ids.append(data["all_user_cross_ids"][i * args.batch_size: (i + 1) * args.batch_size])
        all_user_cross_mask.append(data["all_user_cross_mask"][i * args.batch_size: (i + 1) * args.batch_size])
        all_user_cross_seg.append(data["all_user_cross_seg"][i * args.batch_size: (i + 1) * args.batch_size])
        all_candidate_priors.append(data["all_candidate_priors"][i * args.batch_size: (i + 1) * args.batch_size])
        all_entity_masks.append(data["all_entity_masks"][i * args.batch_size: (i + 1) * args.batch_size])
        batch_train_targets.append(data["all_label"][i * args.batch_size: (i + 1) * args.batch_size])

    for i in range(batch_count):
        logits = model(
            all_ctxt_cross_ids[i],
            all_ctxt_cross_mask[i],
            all_ctxt_cross_seg[i],
            all_cqa_cross_ids[i],
            all_cqa_cross_mask[i],
            all_cqa_cross_seg[i],
            all_topic_cross_ids[i],
            all_topic_cross_mask[i],
            all_topic_cross_seg[i],
            all_user_cross_ids[i],
            all_user_cross_mask[i],
            all_user_cross_seg[i],
            all_candidate_priors[i],
            batch_train_targets[i],
            all_entity_masks[i],
        )

        labels = mindspore.Tensor(batch_train_targets[i], dtype=mindspore.int32)

        temp_eval_accuarcy = tools.accuracy(logits, labels)

        eval_accuarcy += temp_eval_accuarcy
        # total_eval_examples += all_cqa_sent_ids.size(0)

    normalized_eval_accuarcy = eval_accuarcy/LEN_EVAL_SET
    # logger.info(
    #     "the accuarcy of the evaluation set of epoch {} is {}".format(epoch_idx, normalized_eval_accuarcy))
    return normalized_eval_accuarcy


def main(args):
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)
    # # 几张卡
    # nprocs = torch.cuda.device_count()
    # # print(nprocs)
    # args.batch_size = int(args.batch_size / nprocs)

    logger = tools.logger_config(log_path='log.txt', logging_name='lyh')
    logger.info(
        "*" * 32 + "Question-Answer Entity Linking By BERT Start！" + "*" * 32)

    alldata = getalldata(args.data_file)
    # 如果不需要验证集

    LEN_TRAIN_SET = 0
    LEN_EVAL_SET = 0
    LEN_TEST_SET = 0

    if not args.vali:
        train_set, test_set = BuildDataSet.build_train_test(
            alldata, args.split_index)
        logger.info('The train_set size is {:}'.format(len(train_set)))
        logger.info('The test_set size is {:}'.format(len(test_set)))
        LEN_TRAIN_SET = len(train_set)
        LEN_TEST_SET = len(test_set)
    else:
        train_set, vali_set, test_set = BuildDataSet.build_train_vali_test_all(
            alldata, args.split_index, args.top_k)
        logger.info('The train_set size of split_index {} is {:}'.format(args.split_index, len(train_set)))
        logger.info('The vali_set size of split_index {} is {:}'.format(args.split_index, len(vali_set)))
        logger.info('The test_set size of split_index {} is {:}'.format(args.split_index, len(test_set)))
        LEN_TRAIN_SET = len(train_set)
        LEN_EVAL_SET = len(vali_set)
        LEN_TEST_SET = len(test_set)

    # print(train_set[5620]["topic_meta_data"])
    # print(train_set[5620]["user_meta_data"])
    # print(train_set[5620]["cqa_sentences"])

    if args.train_and_eval:
        logger.info('Initing model to train....')
    else:
        logger.info('Loading existing model....')

    model = CQAELModel.Bert4QA(args)

    bert_tokenizer = model.bert_tokenizer
    # bert_tokenizer.add_special_tokens(
    #     {'additional_special_tokens': ["[ENT]"]})
    # model.c_encoder.resize_token_embeddings(len(bert_tokenizer))
    # model.q_encoder.resize_token_embeddings(len(bert_tokenizer))

    # store_path
    result_path = args.model_storage_path
    PATH = os.path.join(
        result_path, 'net_split_{}'.format(args.split_index))
    # if not os.path.exists(result_path):
    #     os.mkdir(result_path)
    # train and eval
    if args.train_and_eval == True:
        # loading training data
        logger.info("Loading training data...")
        train_data = model._process_mentions_for_model(
            train_set, args.top_k, bert_tokenizer, args.max_seq_length, args.max_desc_length, args.max_q_length, args.debug, blink=args.useBLINKDic, logger=logger, 
            num_of_cqa=args.max_cqa_nums, max_user_q_nums=args.max_user_q_nums, max_topic_q_nums=args.max_topic_q_nums, max_topic_nums=args.max_topic_nums,)

        train_dataloader = GeneratorDataset(
            source=SeedDataset(train_data),
            column_names=['all_ctxt_cross_ids', 'all_ctxt_cross_mask', 'all_ctxt_cross_seg', 'all_cqa_cross_ids', 'all_cqa_cross_mask', 'all_cqa_cross_seg', 'all_topic_cross_ids', 'all_topic_cross_mask', 'all_topic_cross_seg', 'all_user_cross_ids', 'all_user_cross_mask', 'all_user_cross_seg', 'all_candidate_priors', 'all_entity_masks', 'all_label'],
            shuffle=True,
            num_parallel_workers=1
        )
        train_dataloader = train_dataloader.batch(batch_size=args.batch_size)

        # loading eval data
        logger.info("Loading validation data...")
        vali_data = model._process_mentions_for_model(
            vali_set, args.top_k, bert_tokenizer, args.max_seq_length, args.max_desc_length, args.max_q_length, args.debug, blink=args.useBLINKDic, logger=logger,
            num_of_cqa=args.max_cqa_nums, max_user_q_nums=args.max_user_q_nums, max_topic_q_nums=args.max_topic_q_nums, max_topic_nums=args.max_topic_nums,
        )

        vali_dataloader = GeneratorDataset(
            source=SeedDataset(vali_data),
            column_names=['all_ctxt_cross_ids', 'all_ctxt_cross_mask', 'all_ctxt_cross_seg', 'all_cqa_cross_ids', 'all_cqa_cross_mask', 'all_cqa_cross_seg', 'all_topic_cross_ids', 'all_topic_cross_mask', 'all_topic_cross_seg', 'all_user_cross_ids', 'all_user_cross_mask', 'all_user_cross_seg', 'all_candidate_priors', 'all_entity_masks', 'all_label'],
            shuffle=False,
            num_parallel_workers=2
        )
        vali_dataloader = vali_dataloader.batch(batch_size=args.batch_size)

        test_data = model._process_mentions_for_model(
            test_set, args.top_k, bert_tokenizer, args.max_seq_length, args.max_desc_length, args.max_q_length, args.debug, blink=args.useBLINKDic, logger=logger,
            num_of_cqa=args.max_cqa_nums, max_user_q_nums=args.max_user_q_nums, max_topic_q_nums=args.max_topic_q_nums, max_topic_nums=args.max_topic_nums,
        )

        test_dataloader = GeneratorDataset(
            source=SeedDataset(test_data),
            column_names=['all_ctxt_cross_ids', 'all_ctxt_cross_mask', 'all_ctxt_cross_seg', 'all_cqa_cross_ids',
                          'all_cqa_cross_mask', 'all_cqa_cross_seg', 'all_topic_cross_ids', 'all_topic_cross_mask',
                          'all_topic_cross_seg', 'all_user_cross_ids', 'all_user_cross_mask', 'all_user_cross_seg',
                          'all_candidate_priors', 'all_entity_masks', 'all_label'],
            shuffle=False,
            num_parallel_workers=2
        )
        test_dataloader = test_dataloader.batch(batch_size=args.batch_size)

        optimizer = nn.SGD(model.get_parameters(), learning_rate=args.learning_rate)

        criterion = nn.CrossEntropyLoss()

        # train_iterator = train_dataloader.create_dict_iterator()

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_set))
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Gradient accumulation steps = %d",
                    args.gradient_accumulation_steps)

        best_vali_epoch_idx = -1
        best_vali_score = -1

        best_test_epoch_idx = -1
        best_test_score = -1

        for epoch_idx in range(args.epochs):
            # train
            train_loss, train_acc = train(
                model, train_data, optimizer, criterion, epoch_idx, logger, args, LEN_TRAIN_SET)
            logger.info('Epoch [{:d}/{:d}], AVG_loss: {:.9f}, Train_ACC: {:.9f}'.format(epoch_idx + 1, args.epochs, train_loss, train_acc))

            # eval
            vali_acc = eval(
                model, vali_data, args, LEN_EVAL_SET)
            logger.info('Epoch [{:d}/{:d}], Vali_ACC: {:.9f}'.format(epoch_idx + 1, args.epochs, vali_acc))
            if vali_acc >= best_vali_score:
                best_vali_score = vali_acc
                best_vali_epoch_idx = epoch_idx + 1
                mindspore.save_checkpoint(model,
                           os.path.join(PATH, "best_model.ckpt"))
                # tools.saveModel(model, tokenizer, PATH_)

            logger.info('Epoch [{:d}/{:d}], now_best_eval_acc:{:.9f}, best_epoch:{:d}\n,'
                        .format(epoch_idx + 1, args.epochs, best_vali_score, best_vali_epoch_idx))

            # add testing
            test_acc = eval(model, test_data, args, LEN_TEST_SET)
            logger.info('Epoch [{:d}/{:d}], Test_ACC: {:.9f}'
                        .format(epoch_idx + 1, args.epochs, test_acc))
            if test_acc >= best_test_score:
                best_test_score = test_acc
                best_test_epoch_idx = epoch_idx + 1
            logger.info('Epoch [{:d}/{:d}], now_best_test_acc:{:.9f}, best_epoch:{:d}\n,'
                        .format(epoch_idx + 1, args.epochs, best_test_score, best_test_epoch_idx))


if __name__ == '__main__':
    t0 = datetime.datetime.now()
    print('='*40)
    print('startTime is {:}'.format(t0))
    for i in range(1):
        parser = argparse.ArgumentParser(
            description='CQA Entity Linking using BERT')
        parser.add_argument("--data_file", type=str,
                            default="../Data/CQAEL_update_dataset_complete.json")
        parser.add_argument("--use_topic", type=bool, default=True)
        parser.add_argument("--use_user", type=bool, default=True)
        parser.add_argument("--use_mixed_qa", type=bool, default=True)
        parser.add_argument("--use_cqa", type=bool, default=True)
        parser.add_argument("--vali", type=bool, default=True)
        parser.add_argument("--split_index", type=int, default=0)
        parser.add_argument("--cuda", type=bool, default=True)
        parser.add_argument("--top_k", type=int, default=20)
        parser.add_argument("--debug", type=bool, default=False)
        parser.add_argument("--max_seq_length", type=int, default=128)
        parser.add_argument("--max_desc_length", type=int, default=128)
        parser.add_argument("--max_q_length", type=int, default=32)
        parser.add_argument("--max_cqa_nums", type=int, default=4)
        parser.add_argument("--max_topic_q_nums", type=int, default=3)
        parser.add_argument("--max_user_q_nums", type=int, default=3)
        parser.add_argument("--max_topic_nums", type=int, default=3)
        parser.add_argument("--batch_size", type=int, default=2)
        parser.add_argument("--warmup_proportion", type=float, default=0.1)
        parser.add_argument("--gradient_accumulation_steps",
                            type=int, default=4)
        parser.add_argument(
            "--print_tr_loss_opt_steps_interval", type=int, default=2)
        parser.add_argument("--max_grad_norm", type=float, default=1.0)
        parser.add_argument("--learning_rate", type=float, default=1e-5)
        parser.add_argument("--epochs", type=int, default=5)
        # parser.add_argument("--clip_grad", type=int, default=1.0)
        parser.add_argument("--out_dim", type=int, default=768)
        parser.add_argument("--add_linear", type=bool, default=False)
        parser.add_argument("--store_file", type=str, default="Logger")
        parser.add_argument("--model_storage_path",
                            type=str, default="../best_model")
        parser.add_argument('--local_rank', default=-1, type=int,
                            help='node rank for distributed training')
        parser.add_argument("--seed", type=int, default=12345)
        parser.add_argument("--train_and_eval", type=bool, default=True)
        parser.add_argument("--useBLINKDic", type=bool, default=True)

        # Transformer-MD hypers
        parser.add_argument("--max_posts", type=int, default=20)
        parser.add_argument("--max_len", type=int, default=32)
        parser.add_argument("--XP", type=bool, default=True)
        parser.add_argument("--start_step", type=int, default=9)
        args = parser.parse_args()
        main(args)
