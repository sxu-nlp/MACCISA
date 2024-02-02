import math
from sklearn.metrics import classification_report
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
import copy
import random

# 超参数
device_index = 1
labels_num = 3
epochs_num = 12
hidden_dim = 2048
learning_rate = 0.00005
device = torch.device('cuda', device_index)
batch_size = 64
demarcation_num = 30432
node_vectors_path = '/home/sxuchatglm/zzz/project/en_embedding.pkl'
picture_vectors_path = '/home/sxuchatglm/zzz/project/pictures_embedding_en.pkl'
dataset_path = '/home/sxuchatglm/zzz/project/data_en.xlsx'
related_nodes_path = '/home/sxuchatglm/zzz/project/one-hop_en.xlsx'

prompt = "重复下面这句话，"

# 读取数据
with open(node_vectors_path, 'rb') as f:
    words_embedding = pickle.load(f)
print(words_embedding.shape)
with open(picture_vectors_path, mode='rb') as f:
    pictures_embedding = []
    while True:
        try:
            obj = pickle.load(f)
            pictures_embedding.append(obj)
        except EOFError:
            break
    print(len(pictures_embedding))

sentence = pd.read_excel(dataset_path, sheet_name='Sheet1')['content'].tolist()

sentence_class = pd.read_excel(dataset_path, sheet_name='Sheet1')['Column2'].tolist()

word_ids = pd.read_excel(related_nodes_path, sheet_name='Sheet1')['词编号'].tolist()

pictures_ids = pd.read_excel(related_nodes_path, sheet_name='Sheet1')['图编号'].tolist()

for i in range(len(pictures_ids)):
    pictures_ids[i] = [int(x) for x in pictures_ids[i].split(',')]
    random.shuffle(pictures_ids[i])

for i in range(len(word_ids)):
    try:
        word_ids[i] = [int(x) for x in word_ids[i].split(',')]
        random.shuffle(word_ids[i])
    except AttributeError:
        word_ids[i] = pictures_ids[i]
# print(pictures_ids[1][2])
# print(word_ids[9])

# 数据集分割
train_sen, test_sen, train_word_ids, test_word_ids, train_picture_ids, test_picture_ids, train_cls, test_cls = train_test_split(
    sentence,
    word_ids,
    pictures_ids,
    sentence_class,
    test_size=0.2,
    random_state=55)
val_sen, test_sen, val_word_ids, test_word_ids, val_picture_ids, test_picture_ids, val_cls, test_cls = train_test_split(
    test_sen,
    test_word_ids,
    test_picture_ids,
    test_cls,
    test_size=0.5,
    random_state=55)
# 将数据打包
train_dataset = list(zip(train_sen, train_word_ids, train_picture_ids, train_cls))
val_dataset = list(zip(val_sen, val_word_ids, val_picture_ids, val_cls))
test_dataset = list(zip(test_sen, test_word_ids, test_picture_ids, test_cls))

# 加载Chatglm模型
tokenizer = AutoTokenizer.from_pretrained("/home/sxuchatglm/ChatGLM-6B-main/chatglm-6b", trust_remote_code=True)
Chatglm = AutoModel.from_pretrained("/home/sxuchatglm/ChatGLM-6B-main/chatglm-6b",
                                    trust_remote_code=True).half().to(device)
Chatglm = Chatglm.eval()


# batch生成
def batch_generator(dataset, batch_size):
    n = len(dataset)
    for i in range(0, n, batch_size):
        batch_data = dataset[i:min(i + batch_size, n)]
        yield batch_data


# 定义分类器
class ChatglmClassifier(nn.Module):
    # 定义构造函数
    def __init__(self):  # 定义类的初始化函数，用户传入的参数
        super(ChatglmClassifier, self).__init__()  # 调用父类nn.module的初始化方法，初始化必要的变量和参数
        self.multi_head_attention1 = torch.nn.MultiheadAttention(embed_dim=2048, num_heads=8, batch_first=True)
        self.multi_head_attention2 = torch.nn.MultiheadAttention(embed_dim=2048, num_heads=8, batch_first=True)
        self.multi_head_attention3 = torch.nn.MultiheadAttention(embed_dim=2048, num_heads=8, batch_first=True)

        self.sigmoid = nn.Sigmoid()

        self.linear_adapter1 = nn.Linear(4096, 4096)
        self.linear_adapter2 = nn.Linear(192, 192)
        self.linear_adapter3 = nn.Linear(2048, 2048)

        self.linear_word = nn.Linear(4096, 2048)
        self.linear_klg = nn.Linear(192, 2048)
        self.linear_klg_plus = nn.Linear(192, 2048)
        self.linear_pic = nn.Linear(2048, 2048)

        self.linear_second = nn.Linear(2048, 2048)
        self.linear_second2 = nn.Linear(2048, 2048)

        self.linear_third = nn.Linear(2048, 2048)
        self.linear_third2 = nn.Linear(2048, 2048)

        self.linear_out = nn.Linear(2048, 3)
        self.linear_final = nn.Linear(768, 3)
        self.LogSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, klgs, glm_sen, glm_sen_musk, klgs_plus, pic_embs):  # 定义forward函数，实现该模块的前向过程
        """
        """
        glm_sen = torch.tanh(self.linear_word(torch.tanh(self.linear_adapter1(glm_sen)) + glm_sen))
        klgs = torch.tanh(self.linear_klg(torch.tanh(self.linear_adapter2(klgs)) + klgs))
        first_att_out, _ = self.multi_head_attention1(klgs, glm_sen, glm_sen, key_padding_mask=glm_sen_musk)

        pic_embs = torch.tanh(self.linear_pic(torch.tanh(self.linear_adapter3(pic_embs)) + pic_embs))
        second_att_out, _ = self.multi_head_attention2(pic_embs, first_att_out, first_att_out)

        second_att_w = self.sigmoid(self.linear_second(second_att_out))
        second_out = first_att_out * (1 - second_att_w) + second_att_w * second_att_out
        klgs_plus = torch.tanh(self.linear_klg_plus(torch.tanh(self.linear_adapter2(klgs_plus)) + klgs_plus))
        third_out, _ = self.multi_head_attention3(klgs_plus, second_out, second_out)

        # third_att_w = self.sigmoid(self.linear_third(third_att_out))
        # second_att_w2 = self.sigmoid(self.linear_third2(second_out))
        # att_w_third = self.sigmoid(third_att_w + second_att_w2)
        # third_out = second_output + att_w_third * third_att_out

        last_in = self.linear_out(third_out).view(-1, 768)
        output = self.linear_final(last_in)
        output = self.LogSoftmax(output)
        return output


model = ChatglmClassifier().to(device)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, epochs_num + 1):
    # 获取数据生成器
    train_batch_data_gen = batch_generator(train_dataset, batch_size)
    val_batch_data_gen = batch_generator(val_dataset, batch_size)
    test_batch_data_gen = batch_generator(test_dataset, batch_size)
    step = 0
    for batch_data in train_batch_data_gen:
        size = len(batch_data)
        step += size
        klgs_all = []
        glm_sen_all = []
        glm_sen_musk_all = []
        klgs_plus_all = []
        pic_embs_all = []
        cls_all = []
        for sentence, word_ids, pic_ids, cls in batch_data:
            klgs = []
            klgs_plus = []
            pic_embs = []
            # while x < 256:
            #     if len(word_ids) <= x < 128:
            #         klgs.append(torch.zeros(192))
            #     elif x < len(ids) and x < 128:
            #         klgs.append(words_embedding[ids[x]])
            #
            #     if x < len(ids):
            #         klgs_plus.append(words_embedding[ids[x]])
            #     elif len(ids) <= x:
            #         klgs_plus.append(torch.zeros(192))
            #
            #     if x < 128 and x < len(ids):
            #         if ids[len(ids) - 1 - x] >= 11386 and x < len(ids):
            #             pic_embs.append(pictures_embedding[ids[x] - 15422])
            #         elif x > len(ids):
            #             pic_embs.append(torch.zeros(2048))
            #     x += 1

            for i in pic_ids:
                pic_embs.append(pictures_embedding[i - demarcation_num])
            if len(pic_embs) > 128:
                pic_embs = pic_embs[0:128]
            else:
                for i in range(len(pic_embs), 128):
                    pic_embs.append(torch.zeros((1, 2048)))

            for i in word_ids:
                klgs.append(words_embedding[i])
            if len(klgs) > 128:
                klgs = klgs[0:128]
            else:
                for i in range(len(klgs), 128):
                    klgs.append(torch.zeros(192))

            if len(word_ids) >= 128:
                klgs_plus = copy.deepcopy(klgs)
                for i in pic_ids:
                    klgs_plus.append(words_embedding[i])
                if len(klgs_plus) > 256:
                    klgs_plus = klgs_plus[0:256]
                else:
                    for i in range(len(klgs_plus), 256):
                        klgs_plus.append(torch.zeros(192))
            elif len(word_ids) < 128:
                klgs_plus = copy.deepcopy(klgs[0:len(word_ids)])
                for i in pic_ids:
                    klgs_plus.append(words_embedding[i])
                if len(klgs_plus) > 256:
                    klgs_plus = klgs_plus[0:256]
                else:
                    for i in range(len(klgs_plus), 256):
                        klgs_plus.append(torch.zeros(192))

            cls_all.append(cls)
            klgs_all.append(klgs)
            klgs_plus_all.append(klgs_plus)
            pic_embs_all.append(pic_embs)
            glm_sen = Chatglm.chat(tokenizer, prompt + sentence, history=[]).to(device)
            if glm_sen.shape[0] < 128:
                glm_sen = torch.cat((glm_sen, torch.zeros((128 - glm_sen.shape[0], 4096)).to(device)), dim=0)
                glm_sen_musk = torch.cat((torch.zeros(glm_sen.shape[0]), torch.ones(128 - glm_sen.shape[0])),
                                         dim=0).type(torch.bool)
            else:
                glm_sen = glm_sen[:128, :]
                glm_sen_musk = torch.zeros(128).type(torch.bool)
            glm_sen_musk_all.append(glm_sen_musk)
            glm_sen_all.append(glm_sen)

        klgs_all = torch.stack([torch.stack(lst) for lst in klgs_all]).to(device)
        klgs_plus_all = torch.stack([torch.stack(lst) for lst in klgs_plus_all]).to(device)
        pic_embs_all = torch.stack([torch.stack(lst) for lst in pic_embs_all]).to(device)
        pic_embs_all = torch.squeeze(pic_embs_all, dim=2)
        glm_sen_musk_all = torch.stack([lst for lst in glm_sen_musk_all]).to(device)
        glm_sen_all = torch.stack([lst for lst in glm_sen_all]).to(device)
        cls_all = torch.tensor(cls_all).to(device)

        output = model(klgs_all, glm_sen_all, glm_sen_musk_all, klgs_plus_all, pic_embs_all)
        loss = criterion(output, cls_all)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 64 == 0:
            print(f"epoch: {epoch}, step:{step}, loss: {loss.item():.4f}")
        print(torch.argmax(output, dim=1).tolist())
        print(cls_all.tolist())

    # 验证集做测试

    pred_all = []
    gold_all = []

    for batch_data in val_batch_data_gen:
        size = len(batch_data)
        step += size
        klgs_all = []
        glm_sen_all = []
        glm_sen_musk_all = []
        klgs_plus_all = []
        pic_embs_all = []
        cls_all = []
        for sentence, word_ids, pic_ids, cls in batch_data:
            klgs = []
            klgs_plus = []
            pic_embs = []
            # while x < 256:
            #     if len(word_ids) <= x < 128:
            #         klgs.append(torch.zeros(192))
            #     elif x < len(ids) and x < 128:
            #         klgs.append(words_embedding[ids[x]])
            #
            #     if x < len(ids):
            #         klgs_plus.append(words_embedding[ids[x]])
            #     elif len(ids) <= x:
            #         klgs_plus.append(torch.zeros(192))
            #
            #     if x < 128 and x < len(ids):
            #         if ids[len(ids) - 1 - x] >= 11386 and x < len(ids):
            #             pic_embs.append(pictures_embedding[ids[x] - 15422])
            #         elif x > len(ids):
            #             pic_embs.append(torch.zeros(2048))
            #     x += 1

            for i in pic_ids:
                pic_embs.append(pictures_embedding[i - demarcation_num])
            if len(pic_embs) > 128:
                pic_embs = pic_embs[0:128]
            else:
                for i in range(len(pic_embs), 128):
                    pic_embs.append(torch.zeros((1, 2048)))

            for i in word_ids:
                klgs.append(words_embedding[i])
            if len(klgs) > 128:
                klgs = klgs[0:128]
            else:
                for i in range(len(klgs), 128):
                    klgs.append(torch.zeros(192))

            if len(word_ids) >= 128:
                klgs_plus = copy.deepcopy(klgs)
                for i in pic_ids:
                    klgs_plus.append(words_embedding[i])
                if len(klgs_plus) > 256:
                    klgs_plus = klgs_plus[0:256]
                else:
                    for i in range(len(klgs_plus), 256):
                        klgs_plus.append(torch.zeros(192))
            elif len(word_ids) < 128:
                klgs_plus = copy.deepcopy(klgs[0:len(word_ids)])
                for i in pic_ids:
                    klgs_plus.append(words_embedding[i])
                if len(klgs_plus) > 256:
                    klgs_plus = klgs_plus[0:256]
                else:
                    for i in range(len(klgs_plus), 256):
                        klgs_plus.append(torch.zeros(192))

            cls_all.append(cls)
            klgs_all.append(klgs)
            klgs_plus_all.append(klgs_plus)
            pic_embs_all.append(pic_embs)
            glm_sen = Chatglm.chat(tokenizer, prompt + sentence, history=[]).to(device)
            if glm_sen.shape[0] < 128:
                glm_sen = torch.cat((glm_sen, torch.zeros((128 - glm_sen.shape[0], 4096)).to(device)), dim=0)
                glm_sen_musk = torch.cat((torch.zeros(glm_sen.shape[0]), torch.ones(128 - glm_sen.shape[0])),
                                         dim=0).type(torch.bool)
            else:
                glm_sen = glm_sen[:128, :]
                glm_sen_musk = torch.zeros(128).type(torch.bool)
            glm_sen_musk_all.append(glm_sen_musk)
            glm_sen_all.append(glm_sen)

        klgs_all = torch.stack([torch.stack(lst) for lst in klgs_all]).to(device)
        klgs_plus_all = torch.stack([torch.stack(lst) for lst in klgs_plus_all]).to(device)
        pic_embs_all = torch.stack([torch.stack(lst) for lst in pic_embs_all]).to(device)
        pic_embs_all = torch.squeeze(pic_embs_all, dim=2)
        glm_sen_musk_all = torch.stack([lst for lst in glm_sen_musk_all]).to(device)
        glm_sen_all = torch.stack([lst for lst in glm_sen_all]).to(device)
        cls_all = torch.tensor(cls_all).to(device)

        with torch.no_grad():
            output = model(klgs_all, glm_sen_all, glm_sen_musk_all, klgs_plus_all, pic_embs_all)

        pred = torch.argmax(output, dim=1).tolist()
        pred_all += pred
        gold_all += cls_all.tolist()
    target_names = ['中性', '积极', '消极']
    report = classification_report(gold_all, pred_all, target_names=target_names, labels=range(len(target_names)),
                                   digits=3)
    print(report)
