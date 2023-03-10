from collections import OrderedDict, Counter  # 排序字典，计数
from tqdm import tqdm  # 进度条
from mindspore import dataset


class BuildDataSet():  # 构建数据集使用，必须继承自Dataset,且前三个函数必须重新定义
    def __init__(self, datalist):
        super(BuildDataSet, self).__init__()
        self.datalist = datalist

    def __len__(self):  # 重构函数用于返回数据长度
        return len(self.datalist)

    def __getitem__(self, item):  # 重构函数用于根据索引返回对应数据
        return self.datalist[item]

    def field_iter(self, field):  # 用于返回数据集中的某列数据
        def get_data():
            for i in range(len(self)):
                yield self[i][field]

        return get_data

    @staticmethod
    def build_train_test(datalist, split_index):  # 用于构建训练及测试集
        train = []
        test = []
        for i in range(len(datalist)):  # 暂定
            if (i + split_index) % 5 == 0:
                test.append(datalist[i])
            else:
                train.append(datalist[i])
        return BuildDataSet(train), BuildDataSet(test)

    @staticmethod
    def build_train_vali_test(datalist, split_index):  # 用于构建训练及测试集
        train = []
        vali = []
        test = []
        val = [4]
        tes = [6, 7]
        for i in range(len(datalist)):  # 暂定
            if (i + split_index) % 10 in tes:
                test.append(datalist[i])
            elif (i + split_index) % 10 in val:
                vali.append(datalist[i])
            else:
                train.append(datalist[i])
        return BuildDataSet(train), BuildDataSet(vali), BuildDataSet(test)

    @staticmethod
    def build_train_vali_test_all(datalist, split_index, top_k):
        train = []
        vali = []
        test = []
        if split_index == 0:
            val = [5]
            tes = [0, 1]
        elif split_index == 1:
            val = [6]
            tes = [2, 3]
        elif split_index == 2:
            val = [8]
            tes = [4, 5]
        elif split_index == 3:
            val = [9]
            tes = [6, 7]
        else:
            val = [7]
            tes = [8, 9]
        # val = [1]
        # tes = [6, 7]
        all_mention_info = []
        for i in range(len(datalist)):
            # 事先处理得到该question下的所有topics
            mention_topics = []
            topics = datalist[i][0]["topics"]
            for topic in topics:
                temp = dict()
                temp['topic_name'] = topic["topic_name"]
                temp['topic_questions'] = topic["topic_question"]
                mention_topics.append(temp)

            # 新建一个列表保存一个question中的所有CQA信息
            cqa_information = []

            mention_num_in_current_question = 0

            cqa_information.append(datalist[i][0]["question_title"])
            mentions = datalist[i][0]["mentions"]
            for mention in mentions:
                if mention["Gold_index"] != -1:
                    mention_num_in_current_question += 1
                    mention_info = {}
                    mention_info["topic_meta_data"] = mention_topics
                    mention_info["user_meta_data"] = []
                    mention_info["qa_sent"] = datalist[i][0]["question_title"]
                    mention_info["mention_set"] = mention["mention"]
                    candi_list = mention["Candidates"].strip().split("\t")
                    mention_info["mention_cand"] = [(
                        candi_list[j], int(candi_list[j + 1]), candi_list[j + 2])
                        for j in range(len(candi_list))[::3]]
                    # if mention["Gold_index"] <= top_k:
                    #     mention_info["mention_target"] = int(
                    #         mention["Gold_index"]) - 1
                    # else:
                    #     mention_info["mention_target"] = 0
                    mention_info["mention_target"] = int(
                        mention["Gold_index"]) - 1
                    all_mention_info.append(mention_info)
            answers = datalist[i][0]["answers"]
            for answer in answers:
                # 只有answer中的mention才有user_meta_data信息，故question中的mention_info["user_meta_data"]设为空列表
                # 在对每个mention进行处理之前先记录下每个answer对应的user_question和user_answer
                user_question = answer["user_question"]
                user_answer = answer["user_answer"]
                cqa_information.append(answer["answer_content"])
                mentions = answer["mentions"]
                for mention in mentions:
                    if mention["Gold_index"] != -1:
                        mention_num_in_current_question += 1
                        mention_info = {}
                        mention_info["topic_meta_data"] = mention_topics
                        mention_info["user_meta_data"] = user_question + \
                            user_answer
                        mention_info["qa_sent"] = answer["answer_content"]
                        mention_info["mention_set"] = mention["mention"]
                        candi_list = mention["Candidates"].strip().split("\t")
                        mention_info["mention_cand"] = [(
                            candi_list[j], int(candi_list[j + 1]), candi_list[j + 2])
                            for j in range(len(candi_list))[::3]]
                        # if mention["Gold_index"] <= top_k:
                        #     mention_info["mention_target"] = int(
                        #         mention["Gold_index"]) - 1
                        # else:
                        #     mention_info["mention_target"] = 0
                        mention_info["mention_target"] = int(
                            mention["Gold_index"]) - 1
                        all_mention_info.append(mention_info)

            for k in range(len(all_mention_info)-1, len(all_mention_info)-1-mention_num_in_current_question, -1):
                all_mention_info[k]["cqa_sentences"] = cqa_information
        # mention_info: {"qa_sent": "Why is there a shortage of qualified workers in the United States?", "mention_set": "the United States", "Candidates": [(United States, 3434750, 0.4127906976744186), ....]
        # , "mention_target": 1}

        # for i in range(len(all_mention_info)):
        #     if (i + split_index) % 10 in tes:
        #         test.append(all_mention_info[i])
        #     elif (i + split_index) % 10 in val:
        #         vali.append(all_mention_info[i])
        #     else:
        #         train.append(all_mention_info[i])

        for i in range(len(all_mention_info)):
            if (i) % 10 in tes:
                test.append(all_mention_info[i])
            elif (i) % 10 in val:
                vali.append(all_mention_info[i])
            else:
                train.append(all_mention_info[i])

        return BuildDataSet(train), BuildDataSet(vali), BuildDataSet(test)

    @ staticmethod
    def build_all(datalist):
        all_mention_info = []
        for i in range(len(datalist)):
            mentions = datalist[i][0]["mentions"]
            for mention in mentions:
                if mention["Gold_index"] != -1:
                    mention_info = {}
                    mention_info["qa_sent"] = datalist[i][0]["question_title"]
                    mention_info["mention_set"] = mention["mention"]
                    candi_list = mention["Candidates"].strip().split("\t")
                    mention_info["mention_cand"] = [(
                        candi_list[j], int(candi_list[j + 1]), candi_list[j + 2])
                        for j in range(len(candi_list))[::3]]
                    mention_info["mention_target"] = (mention["Gold_index"])
                    all_mention_info.append(mention_info)
            answers = datalist[i][0]["answers"]
            for answer in answers:
                mentions = answer["mentions"]
                for mention in mentions:
                    if mention["Gold_index"] != -1:
                        mention_info = {}
                        mention_info["qa_sent"] = answer["answer_content"]
                        mention_info["mention_set"] = mention["mention"]
                        candi_list = mention["Candidates"].strip().split("\t")
                        mention_info["mention_cand"] = [(
                            candi_list[j], int(candi_list[j + 1]), candi_list[j + 2])
                            for j in range(len(candi_list))[::3]]
                        mention_info["mention_target"] = (
                            mention["Gold_index"])
                        all_mention_info.append(mention_info)
        return BuildDataSet(all_mention_info)
