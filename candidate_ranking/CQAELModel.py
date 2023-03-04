from transformers import BertTokenizer, BertConfig
from tqdm import tqdm
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer, One
from bert_model import BertModel
import os
import sys
import heapq
import Levenshtein
import mindspore
import difflib
import numpy as np

sys.path.append(os.path.abspath(".."))
# from xlnet_files.models.xlnet.modeling_xlnet import XLNetModel, XLNetConfig

class Bert4QA(nn.Cell):
    def __init__(self, args):
        super(Bert4QA, self).__init__()
        self.top_k = args.top_k
        self.use_QA = args.use_mixed_qa
        self.use_topic = args.use_topic
        self.use_user = args.use_user
        # self-attention on QA_sentences
        self.use_cqa = args.use_cqa

        # BERT tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')

        config = BertConfig.from_pretrained('../Data/bert-base-uncased')
        config.dtype = mindspore.dtype.float32
        config.compute_type = mindspore.dtype.float16

        # context encoder
        self.c_encoder = BertModel(config, is_training=True, use_one_hot_embeddings=False)
        self.weight_init = TruncatedNormal(config.initializer_range)
        
        # aux data encoder
        self.q_encoder = BertModel(config, is_training=True, use_one_hot_embeddings=False)

        self.max_topic_nums = args.max_topic_nums
        self.max_topic_q_nums = args.max_topic_q_nums
        self.max_user_q_nums = args.max_user_q_nums
        self.max_seq_length = args.max_seq_length
        self.max_q_length = args.max_q_length
        self.max_cqa_nums = args.max_cqa_nums

        self.classifier_ctxt = nn.Dense(768, 1)
        self.classifier_ques = nn.Dense(768, 1)

        # self.scoreLinear = nn.Dense(5, 1, bias=False)
        weight = mindspore.Tensor(np.ones([1, 5]), dtype=mindspore.dtype.float32)
        self.scoreLinear = nn.Dense(5, 1, weight_init=weight)

        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dropout = nn.Dropout(1 - 0.1)
        self.lossFun = nn.CrossEntropyLoss()

    def construct(
            self,
            all_ctxt_cross_ids=None,
            all_ctxt_cross_mask=None,
            all_ctxt_cross_seg=None,
            all_cqa_cross_ids=None,
            all_cqa_cross_mask=None,
            all_cqa_cross_seg=None,
            all_topic_cross_ids=None,
            all_topic_cross_mask=None,
            all_topic_cross_seg=None,
            all_user_cross_ids=None,
            all_user_cross_mask=None,
            all_user_cross_seg=None,
            candidate_priors=None,
            labels=None,
            entity_mask=None,
    ):
        # 5 features (base + cqa + topic + user)
        if self.use_QA and self.use_user and self.use_topic and self.use_cqa:
            # ctxt_scores
            bsz, _, qa_max_len = all_ctxt_cross_ids.shape
            flat_input_ctxt_ids = all_ctxt_cross_ids.view(-1, qa_max_len)
            flat_input_ctxt_mask = all_ctxt_cross_mask.view(-1, qa_max_len)
            flat_input_ctxt_seg = all_ctxt_cross_seg.view(-1, qa_max_len)

            ctxt_output = self.c_encoder(flat_input_ctxt_ids,flat_input_ctxt_seg, flat_input_ctxt_mask,)

            # (bsz*20*5) * 768 -> bsz * 20 * 768
            ctxt_embs = ctxt_output[0][:, 0].view(bsz, -1, 768)
            # bsz * 20 * 5 * 1
            ctxt_scores = self.classifier_ctxt(ctxt_embs)
            # bsz * 20
            ctxt_scores = mindspore.ops.squeeze(ctxt_scores, axis=2)

            # cqa_scores
            bsz, _, qa_max_len = all_cqa_cross_ids.shape
            flat_input_cqa_ids = all_cqa_cross_ids.view(-1, qa_max_len)
            flat_input_cqa_mask = all_cqa_cross_mask.view(-1, qa_max_len)
            flat_input_cqa_seg = all_cqa_cross_seg.view(-1, qa_max_len)

            cqa_output = self.q_encoder(flat_input_cqa_ids,flat_input_cqa_seg, flat_input_cqa_mask,)

            # (bsz*20*5) * 768 -> bsz * 20 * 768
            cqa_embs = cqa_output[0][:, 0].view(bsz, -1, 768)
            # bsz * 20 * 5 * 1
            cqa_scores = self.classifier_ques(cqa_embs)
            # bsz * 20
            # cqa_scores = mindspore.ops.unsqueeze(cqa_scores, dim=2)

            # topic_scores
            bsz, _, qa_max_len = all_topic_cross_ids.shape
            flat_input_topic_ids = all_topic_cross_ids.view(-1, qa_max_len)
            flat_input_topic_mask = all_topic_cross_mask.view(-1, qa_max_len)
            flat_input_topic_seg = all_topic_cross_seg.view(-1, qa_max_len)

            topic_output = self.q_encoder(flat_input_topic_ids,flat_input_topic_seg, flat_input_topic_mask,)

            # (bsz*20*5) * 768 -> bsz * 20 * 768
            topic_embs = topic_output[0][:, 0].view(bsz, -1, 768)
            # bsz * 20 * 5 * 1
            topic_scores = self.classifier_ques(topic_embs)
            # bsz * 20
            # topic_scores = mindspore.ops.unsqueeze(topic_scores, dim=2)

            # user_scores
            bsz, _, qa_max_len = all_user_cross_ids.shape
            flat_input_user_ids = all_user_cross_ids.view(-1, qa_max_len)
            flat_input_user_mask = all_user_cross_mask.view(-1, qa_max_len)
            flat_input_user_seg = all_user_cross_seg.view(-1, qa_max_len)

            user_output = self.q_encoder(flat_input_user_ids,flat_input_user_seg, flat_input_user_mask,)

            # (bsz*20*5) * 768 -> bsz * 20 * 768
            user_embs = user_output[0][:, 0].view(bsz, -1, 768)
            # bsz * 20 * 5 * 1
            user_scores = self.classifier_ques(user_embs)
            # bsz * 20
            # user_scores = mindspore.ops.unsqueeze(user_scores, dim=2)

            final_score_vec = mindspore.ops.stack(
                (ctxt_scores, candidate_priors), axis=2)
            final_score_vec = mindspore.ops.concat(
                (final_score_vec, cqa_scores), 2)
            final_score_vec = mindspore.ops.concat(
                (final_score_vec, topic_scores), 2)
            final_score_vec = mindspore.ops.concat(
                (final_score_vec, user_scores), 2)
            final_score = self.scoreLinear(final_score_vec)
            reshaped_logits = mindspore.ops.squeeze(final_score, axis=2)

            entity_mask = (1.0 - entity_mask) * -1000
            reshaped_logits = reshaped_logits + entity_mask

            outputs = reshaped_logits
            # loss = self.lossFun(reshaped_logits, labels)
            # outputs = (loss,) + outputs

            return outputs

    @staticmethod
    def _select_field(samples, field):
        """返回一个列表，里面包含所有候选实体的信息"""
        return [
            [cand[field] for cand in sample["candidate_features"]] for sample in samples
        ]

    @staticmethod
    def _get_candidate_tokens_representation(
            candidate_prior,
            bert_tokenizer,
            candidate_title,
            candidate_desc,
            sample,
            max_seq_length,
            max_desc_length,
            # 32?
            max_q_length,
            num_of_cqa,
            num_of_user_q,
            num_of_topic_q,
            num_of_topic,
    ):
        # ctxt cross_ids
        max_sub_seq_length = (max_seq_length - 3) // 2
        candidate_title_tokens = bert_tokenizer.tokenize(candidate_title)
        candidate_desc_tokens = bert_tokenizer.tokenize(candidate_desc)
        max_desc_seq_length = max_sub_seq_length - len(candidate_title_tokens)
        if len(candidate_desc_tokens) > max_desc_seq_length:
            candidate_desc_tokens = candidate_desc_tokens[:max_desc_seq_length]
        cand_tokens = (candidate_title_tokens + ["[ENT]"] + candidate_desc_tokens)
        context = sample["qa_sent"]
        context_tokens = bert_tokenizer.tokenize(context)
        if len(context_tokens) > max_sub_seq_length:
            context_tokens = context_tokens[:max_sub_seq_length]

        # 构造cross-encoder tokens
        tokens = (["[CLS]"] + context_tokens + ["[SEP]"] +
                  cand_tokens + ["[SEP]"])
        ctxt_cross_seg = [0] * (len(context_tokens) + 2) + \
            [1] * (len(cand_tokens) + 1)
        ctxt_cross_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
        ctxt_cross_mask = [1] * len(ctxt_cross_ids)

        # 如果不够最大长度，将input_ids, input_mask, 和segment_ids都填充zero-padding
        padding = [0] * (max_seq_length - len(ctxt_cross_ids))
        ctxt_cross_ids += padding
        ctxt_cross_mask += padding
        ctxt_cross_seg += padding

        assert len(ctxt_cross_ids) == max_seq_length
        assert len(ctxt_cross_mask) == max_seq_length
        assert len(ctxt_cross_seg) == max_seq_length

        # cqa cross_ids
        max_sub_seq_length = (max_desc_length - 3) // 2
        candidate_title_tokens = bert_tokenizer.tokenize(candidate_title)
        candidate_desc_tokens = bert_tokenizer.tokenize(candidate_desc)
        max_desc_seq_length = max_sub_seq_length - len(candidate_title_tokens)
        if len(candidate_desc_tokens) > max_desc_seq_length:
            candidate_desc_tokens = candidate_desc_tokens[:max_desc_seq_length]
        cand_tokens = (candidate_title_tokens + ["[ENT]"] + candidate_desc_tokens)

        all_cqa = []
        for cqa_sent in sample["cqa_sentences"]:
            all_cqa.append(cqa_sent)

        context = sample["qa_sent"]
        string_simi_scores = []
        for ques in all_cqa:
            # three types of string similarity score
            score1 = Levenshtein.ratio(context, ques)
            score2 = Levenshtein.jaro_winkler(context, ques)
            score3 = difflib.SequenceMatcher(context, ques).ratio()
            score_avg = (score1 + score2 + score3) / 3
            string_simi_scores.append(score_avg)
        index = heapq.nlargest(num_of_cqa, range(len(string_simi_scores)), string_simi_scores.__getitem__)

        cqa_context = ""
        for i in index:
            cqa_context += all_cqa[i]
        context_tokens = bert_tokenizer.tokenize(cqa_context)
        if len(context_tokens) > max_sub_seq_length:
            context_tokens = context_tokens[:max_sub_seq_length]

        tokens = (["[CLS]"] + cand_tokens + ["[SEP]"] +
                  context_tokens + ["[SEP]"])
        cqa_cross_seg = [0] * (len(cand_tokens) + 2) + \
            [1] * (len(context_tokens) + 1)
        cqa_cross_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
        cqa_cross_mask = [1] * len(cqa_cross_ids)

        padding = [0] * (max_seq_length - len(cqa_cross_ids))
        cqa_cross_ids += padding
        cqa_cross_mask += padding
        cqa_cross_seg += padding

        assert len(cqa_cross_ids) == max_seq_length
        assert len(cqa_cross_mask) == max_seq_length
        assert len(cqa_cross_seg) == max_seq_length


        # topic cross_ids
        max_sub_seq_length = (max_desc_length - 3) // 2
        candidate_title_tokens = bert_tokenizer.tokenize(candidate_title)
        candidate_desc_tokens = bert_tokenizer.tokenize(candidate_desc)
        max_desc_seq_length = max_sub_seq_length - len(candidate_title_tokens)
        if len(candidate_desc_tokens) > max_desc_seq_length:
            candidate_desc_tokens = candidate_desc_tokens[:max_desc_seq_length]
        cand_tokens = (candidate_title_tokens + ["[ENT]"] + candidate_desc_tokens)

        all_topic_ques = []
        for topic in sample["topic_meta_data"]:
            topic_name = topic["topic_name"]
            for q in topic["topic_questions"]:
                ques = topic_name + q
                all_topic_ques.append(ques)

        context = sample["qa_sent"]
        string_simi_scores = []
        for ques in all_topic_ques:
            # three types of string similarity score
            score1 = Levenshtein.ratio(context, ques)
            score2 = Levenshtein.jaro_winkler(context, ques)
            score3 = difflib.SequenceMatcher(context, ques).ratio()
            score_avg = (score1 + score2 + score3) / 3
            string_simi_scores.append(score_avg)
        index = heapq.nlargest(num_of_topic_q, range(len(string_simi_scores)), string_simi_scores.__getitem__)

        topic_context = ""
        for i in index:
            topic_context += all_topic_ques[i]
        context_tokens = bert_tokenizer.tokenize(topic_context)
        if len(context_tokens) > max_sub_seq_length:
            context_tokens = context_tokens[:max_sub_seq_length]

        tokens = (["[CLS]"] + cand_tokens + ["[SEP]"] +
                  context_tokens + ["[SEP]"])
        topic_cross_seg = [0] * (len(cand_tokens) + 2) + \
            [1] * (len(context_tokens) + 1)
        topic_cross_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
        topic_cross_mask = [1] * len(topic_cross_ids)

        padding = [0] * (max_seq_length - len(topic_cross_ids))
        topic_cross_ids += padding
        topic_cross_mask += padding
        topic_cross_seg += padding

        assert len(topic_cross_ids) == max_seq_length
        assert len(topic_cross_mask) == max_seq_length
        assert len(topic_cross_seg) == max_seq_length


        # user cross_ids
        max_sub_seq_length = (max_desc_length - 3) // 2
        candidate_title_tokens = bert_tokenizer.tokenize(candidate_title)
        candidate_desc_tokens = bert_tokenizer.tokenize(candidate_desc)
        max_desc_seq_length = max_sub_seq_length - len(candidate_title_tokens)
        if len(candidate_desc_tokens) > max_desc_seq_length:
            candidate_desc_tokens = candidate_desc_tokens[:max_desc_seq_length]
        cand_tokens = (candidate_title_tokens + ["[ENT]"] + candidate_desc_tokens)

        all_user_ques = []
        for q in sample["user_meta_data"]:
            ques = q
            all_user_ques.append(ques)

        context = sample["qa_sent"]
        string_simi_scores = []
        for ques in all_user_ques:
            # three types of string similarity score
            score1 = Levenshtein.ratio(context, ques)
            score2 = Levenshtein.jaro_winkler(context, ques)
            score3 = difflib.SequenceMatcher(context, ques).ratio()
            score_avg = (score1 + score2 + score3) / 3
            string_simi_scores.append(score_avg)
        index = heapq.nlargest(num_of_user_q, range(len(string_simi_scores)), string_simi_scores.__getitem__)

        user_context = ""
        for i in index:
            user_context += all_user_ques[i]
        context_tokens = bert_tokenizer.tokenize(user_context)
        if len(context_tokens) > max_sub_seq_length:
            context_tokens = context_tokens[:max_sub_seq_length]

        tokens = (["[CLS]"] + cand_tokens + ["[SEP]"] +
                  context_tokens + ["[SEP]"])
        user_cross_seg = [0] * (len(cand_tokens) + 2) + \
            [1] * (len(context_tokens) + 1)
        user_cross_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
        user_cross_mask = [1] * len(user_cross_ids)

        padding = [0] * (max_seq_length - len(user_cross_ids))
        user_cross_ids += padding
        user_cross_mask += padding
        user_cross_seg += padding

        assert len(user_cross_ids) == max_seq_length
        assert len(user_cross_mask) == max_seq_length
        assert len(user_cross_seg) == max_seq_length

        return {
            "ctxt_cross_ids": ctxt_cross_ids,
            "ctxt_cross_mask": ctxt_cross_mask,
            "ctxt_cross_seg": ctxt_cross_seg,
            "cqa_cross_ids": cqa_cross_ids,
            "cqa_cross_mask": cqa_cross_mask,
            "cqa_cross_seg": cqa_cross_seg,
            "topic_cross_ids": topic_cross_ids,
            "topic_cross_mask": topic_cross_mask,
            "topic_cross_seg": topic_cross_seg,
            "user_cross_ids": user_cross_ids,
            "user_cross_mask": user_cross_mask,
            "user_cross_seg": user_cross_seg,
            "candidate_prior": float(candidate_prior),
        }


    @staticmethod
    def _process_mentions_for_model(
            mentions,
            top_k,
            bert_tokenizer,
            max_seq_length,
            max_desc_length,
            max_q_length,
            debug=False,
            silent=False,
            Training=True,
            blink=True,
            logger=None,
            num_of_cqa=None,
            max_user_q_nums=None,
            max_topic_q_nums=None,
            max_topic_nums=None,
    ):
        processed_mentions = []

        if debug:
            mentions = mentions[:50]

        if silent:
            iter_ = mentions
        else:
            iter_ = tqdm(mentions)

        # 用来统计没有实体描述的候选实体个数
        num_candidate_without_des = 0
        # 用来统计候选实体的总个数
        num_total_candidate = 0

        # get list of entities
        entities = []
        # use blink dictionary
        if blink:
            fin = open("../Data/Entity_id_description_blink",
                       "r", encoding='utf-8')
        # if not blink
        else:
            fin = open("../Data/Entity_id_description", "r", encoding='utf-8')

        for line in fin.readlines():
            entity = line.split("\t")
            entities.append(entity)

        num_candidate_used_for_training = 0

        # 对每一条mention进行处理
        for idx, mention in enumerate(iter_):
            if Training:
                if mention["mention_target"] >= top_k:
                    num_candidate_used_for_training += 1
                    continue
            # context_features = Bert4QA._get_context_tokens_representation(
            #     xlnet_tokenizer, mention, max_seq_length)
            candidates = mention["mention_cand"]

            # 用来存放top_k个candidate的token representation
            candidate_features = []

            for candidate in candidates[:top_k]:
                flag = 0
                for entity in entities:
                    if str(candidate[1]).strip() == entity[1].strip():
                        candidate_desc = entity[2]
                        flag = 1
                        break
                if flag == 0:
                    candidate_desc = ""
                    num_candidate_without_des += 1
                num_total_candidate += 1
                # candidate_desc, flag = Bert4QA._get_condidate_description(
                #     candidate, title_token)
                # num_total_candidate += 1
                # if flag == 0:
                #     num_candidate_without_des += 1
                candidate_title = candidate[0]
                candidate_obj = Bert4QA._get_candidate_tokens_representation(
                    candidate[2],
                    bert_tokenizer,
                    candidate_title,
                    candidate_desc,
                    mention,
                    max_seq_length,
                    max_desc_length,
                    max_q_length,
                    num_of_cqa,
                    max_user_q_nums,
                    max_topic_q_nums,
                    max_topic_nums,
                )
                # print(candidate_obj)
                candidate_features.append(candidate_obj)

            # entity_mask 表明当前mention的有效候选实体数量
            entity_mask = [1] * len(candidate_features) + \
                          [0] * (top_k - len(candidate_features))

            # 如果当前mention数量小于topk，填充候选实体至topk
            if len(candidate_features) < top_k:
                candidate_title = ""
                candidate_desc = ""
                padding_candidate_obj = Bert4QA._get_candidate_tokens_representation(
                    0,
                    bert_tokenizer,
                    candidate_title,
                    candidate_desc,
                    mention,
                    max_seq_length,
                    max_desc_length,
                    max_q_length,
                    num_of_cqa,
                    max_user_q_nums,
                    max_topic_q_nums,
                    max_topic_nums,
                )
                for _ in range(top_k - len(candidate_features)):
                    candidate_features.append(padding_candidate_obj)

            # 加两个断言，看构造出来的mention是否符合要求
            assert len(candidate_features) == top_k
            assert len(entity_mask) == top_k

            # 处理label，因为在DataProcess已经-1处理过了，所以这里不需要-1
            label = mention["mention_target"]
            processed_mentions.append(
                {
                    "candidate_features": candidate_features,
                    "label": label,
                    "entity_mask": entity_mask,
                }
            )

        # 打印candidate的信息
        logger.info("total number of candidates is {}".format(
            num_total_candidate))
        logger.info("candidates_without_desc is {}".format(
            num_candidate_without_des))
        logger.info("number of mentions that are filtered for training is {}".format(
            num_candidate_used_for_training))

        # 处理成tensor形式
        all_ctxt_cross_ids = mindspore.Tensor(
            Bert4QA._select_field(processed_mentions, "ctxt_cross_ids")
        ).long()
        all_ctxt_cross_mask = mindspore.Tensor(
            Bert4QA._select_field(processed_mentions, "ctxt_cross_mask")
        ).long()
        all_ctxt_cross_seg = mindspore.Tensor(
            Bert4QA._select_field(processed_mentions, "ctxt_cross_seg")
        ).long()
        all_cqa_cross_ids = mindspore.Tensor(
            Bert4QA._select_field(processed_mentions, "cqa_cross_ids")
        ).long()
        all_cqa_cross_mask = mindspore.Tensor(
            Bert4QA._select_field(processed_mentions, "cqa_cross_mask")
        ).long()
        all_cqa_cross_seg = mindspore.Tensor(
            Bert4QA._select_field(processed_mentions, "cqa_cross_seg")
        ).long()
        all_topic_cross_ids = mindspore.Tensor(
            Bert4QA._select_field(processed_mentions, "topic_cross_ids")
        ).long()
        all_topic_cross_mask = mindspore.Tensor(
            Bert4QA._select_field(processed_mentions, "topic_cross_mask")
        ).long()
        all_topic_cross_seg = mindspore.Tensor(
            Bert4QA._select_field(processed_mentions, "topic_cross_seg")
        ).long()
        all_user_cross_ids = mindspore.Tensor(
            Bert4QA._select_field(processed_mentions, "user_cross_ids"),
            dtype=mindspore.int32,
        )
        all_user_cross_mask = mindspore.Tensor(
            Bert4QA._select_field(processed_mentions, "user_cross_mask")
        ).long()
        all_user_cross_seg = mindspore.Tensor(
            Bert4QA._select_field(processed_mentions, "user_cross_seg")
        ).long()
        all_candidate_priors = mindspore.Tensor(
            Bert4QA._select_field(processed_mentions, "candidate_prior"),
        )
        all_entity_masks = mindspore.Tensor(
            [s["entity_mask"] for s in processed_mentions]
        ).long()
        all_label = mindspore.Tensor(
            [s["label"] for s in processed_mentions]
        ).long()


        data = {
            "all_ctxt_cross_ids": all_ctxt_cross_ids,
            "all_ctxt_cross_mask": all_ctxt_cross_mask,
            "all_ctxt_cross_seg": all_ctxt_cross_seg,
            "all_cqa_cross_ids": all_cqa_cross_ids,
            "all_cqa_cross_mask": all_cqa_cross_mask,
            "all_cqa_cross_seg": all_cqa_cross_seg,
            "all_topic_cross_ids": all_topic_cross_ids,
            "all_topic_cross_mask": all_topic_cross_mask,
            "all_topic_cross_seg": all_topic_cross_seg,
            "all_user_cross_ids": all_user_cross_ids,
            "all_user_cross_mask": all_user_cross_mask,
            "all_user_cross_seg": all_user_cross_seg,
            "all_candidate_priors": all_candidate_priors,
            "all_entity_masks": all_entity_masks,
            "all_label": all_label,
        }

        # tensor_data = TensorDataset(
        #     all_ctxt_cross_ids,
        #     all_ctxt_cross_mask,
        #     all_ctxt_cross_seg,
        #
        #     all_cqa_cross_ids,
        #
        #     all_candidate_priors,
        #     all_label,
        #     all_entity_masks,
        #
        #     all_topic_cross_ids,
        #     all_user_cross_ids,
        #     all_uesr_flag,
        # )

        if logger is not None:
            logger.info("all_ctxt_cross_ids shape:{}".format(
                all_ctxt_cross_ids.shape))
            logger.info("all_ctxt_cross_mask shape:{}".format(
                all_ctxt_cross_mask.shape))
            logger.info("all_ctxt_cross_seg shape:{}".format(
                all_ctxt_cross_seg.shape))

            logger.info("all_cqa_cross_ids shape:{}".format(
                all_cqa_cross_ids.shape))
            logger.info("all_cqa_cross_mask shape:{}".format(
                all_cqa_cross_ids.shape))
            logger.info("all_cqa_cross_seg shape:{}".format(
                all_cqa_cross_ids.shape))
            
            logger.info("all_candidate_priors shape:{}".format(
                all_candidate_priors.shape))
            logger.info("all_entity_masks shape:{}".format(
                all_entity_masks.shape))
            logger.info("all_label shape:{}".format(all_label.shape))

            logger.info("all_topic_cross_ids shape:{}".format(all_topic_cross_ids.shape))
            logger.info("all_topic_cross_mask shape:{}".format(all_topic_cross_mask.shape))
            logger.info("all_topic_cross_seg shape:{}".format(all_topic_cross_seg.shape))

            logger.info("all_user_cross_ids shape:{}".format(all_user_cross_ids.shape))
            logger.info("all_user_cross_mask shape:{}".format(all_user_cross_mask.shape))
            logger.info("all_user_cross_seg shape:{}".format(all_user_cross_seg.shape))
            
                
        return data
