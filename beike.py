import os
import csv
import torch
import transformers
import numpy as np
import pandas as pd
from transformers import *
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

# 定义device，tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('D:\\NLP相关项目\\关系抽取项目实战\\bert-base-chinese')
# 定义模型保存文件
output_dir = './models/'
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

best_score = 0
batch_size = 128

# 读取训练数据
train_question = open('./train.query.tsv','r',encoding = 'utf-8').readlines()
train_answer = open('train.reply.tsv','r',encoding='utf-8').readlines()

# 读取测试数据
test_question = open('./test.query.tsv','r',encoding = 'utf-8').readlines()
test_answer = open('./test.reply.tsv','r',encoding='utf-8').readlines()

# 数据预处理，获取文本及标签
def data_preprocess(question_data,answer_data):
    text = []
    label = []
    for question in question_data:
        question_ids = question.split()[0]
        question_text = question.split()[1]
        for reply in answer_data:
            content = [question_text]
            if question_ids == reply.split()[0]:
                reply_text = reply.split()[2]
                content.append(reply_text)
                text.append(content)
                label.append(int(reply.split()[-1]))
    return text,label

def convert(names, train_text, target): # name_list, sentence_list, label_list
    input_ids, token_type_ids, attention_mask = [], [], []
    for content in train_text:
        encoded_dict = tokenizer(
            content[0],        # 输入文本
            content[1],
            add_special_tokens = True,      # 添加 '[CLS]' 和 '[SEP]'
            max_length = 64,             # 不够填充
            padding = 'max_length',
            truncation = True,             # 太长截断
            return_tensors = 'pt',         # 返回 pytorch tensors 格式的数据
        )
        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        attention_mask.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)

    input_ids = torch.LongTensor(input_ids)
    token_type_ids = torch.LongTensor(token_type_ids)
    attention_mask = torch.LongTensor(attention_mask)
    target = torch.LongTensor(target)

    return input_ids, token_type_ids, attention_mask, target

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten() # [3, 5, 8, 1, 2, ....]
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def save(model):
    # save
    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)

def eval(model, validation_dataloader):
    model.eval()
    eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2])[0]
            logits = logits.detach().cpu().numpy()
            label_ids = batch[3].cpu().numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    global best_score
    if best_score < eval_accuracy / nb_eval_steps:
        best_score = eval_accuracy / nb_eval_steps
        save(model)


def train_eval():
    input_ids, token_type_ids, attention_mask, labels = convert([], train_text, train_label)

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, random_state=1,
                                                                          test_size=0.1)
    train_token, val_token, _, _ = train_test_split(token_type_ids, labels, random_state=1, test_size=0.1)
    train_mask, val_mask, _, _ = train_test_split(attention_mask, labels, random_state=1, test_size=0.1)

    train_data = Data.TensorDataset(train_inputs, train_token, train_mask, train_labels)
    train_dataloader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    validation_data = Data.TensorDataset(val_inputs, val_token, val_mask, val_labels)
    validation_dataloader = Data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained('D:\\NLP相关项目\\关系抽取项目实战\\bert-base-chinese',
                                                               num_labels=2).to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    for _ in range(2):
        for i, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            loss = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2], labels=batch[3])[0]
            print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                eval(model, validation_dataloader)

