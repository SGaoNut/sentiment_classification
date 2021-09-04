# -*- coding: utf-8 -*-

"""
@author: shan
@software: PyCharm
@file: Bert_sequence_classification.py
@time: 2021/8/28 8:13 下午
"""
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings('ignore')

train_df = pd.read_csv('./dataset/train.csv')
val_df = pd.read_csv('./dataset/val.csv')
test_df = pd.read_csv('./dataset/test.csv')

print(val_df.head())
"""
Hugging Face Transformers 

Transformers提供了NLP领域大量state-of-art的预训练语言模型结构的模型和调用框架。  
到目前为止，transformers 提供了超过100种语言的，32种预训练语言模型，简单，强大，高性能，是新手入门的不二选择。  
### BERT 输入格式


### BERT 文本分类输入


### 使用TFBertForSequenceClassification进行文本分类
https://huggingface.co/transformers/model_doc/bert.html#tfbertforsequenceclassification   

BERT 具有两种输出  
1. pooler output，对应的[CLS]的输出   
2. sequence output，对应的是序列中的所有字的最后一层hidden输出last_hidden_state。   

BERT主要可以处理两种，
- 一种任务是分类/回归任务（使用的是pooler output）
- 一种是序列任务（sequence output）。  
TFBertForSequenceClassification，即使用pooler output接softmax进行分类任务。  

"""

from transformers import BertTokenizer

# 定义bert-base-chinese的tokenzier

tokenizers = BertTokenizer.from_pretrained('bert-base-chinese')

test_sentence = '写在年末冬初孩子流感的第五天，我们仍然没有忘记热情拥抱这2020年的第一天。'
# 使用bert-base-chinese的tokenizer将文本转化为bert的输入

bert_input = tokenizers.encode_plus(
    test_sentence,
    add_special_tokens=True,  # 标记是否添加[CLS], [SEP]特殊字符
    max_length=50,  # 最长序列长度
    pad_to_max_length=True,  # 标记是否添加[PAD]到最长长度
    truncation=True,  # 标记是否截断
    return_attention_mask=True,  # 添加注意力掩码，使注意力计算不关注pad的数据
)

for k, v in bert_input.items():
    print(k)
    print(v)


def sample2feature(text, max_length):
    return tokenizers.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
    )


def map_sample2dict(input_ids, token_type_ids, attention_masks, label):
    return {
               "input_ids": input_ids,
               "token_type_ids": token_type_ids,
               "attention_mask": attention_masks
           }, label


# 创建TF数据集
def build_dataset(df, max_length):
    # 准备列表，以便我们可以从列表中构建最终的TensorFlow数据集
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    # 将输入数据转化为BERT输入
    for _, row in df.iterrows():
        text, label = row["text"], row["label"]
        bert_input = sample2feature(text, max_length)  # 对文本进行转换成BERT输入
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    dataset = tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list))
    dataset = dataset.map(map_sample2dict)
    return dataset


BATCH_SIZE = 32
MAX_SEQ_LEN = 240
NUM_LABELS = 3
BUFFER_SIZE = len(train_df)
train_dataset = build_dataset(train_df, MAX_SEQ_LEN). \
    shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(BUFFER_SIZE)
val_dataset = build_dataset(val_df, MAX_SEQ_LEN).batch(BATCH_SIZE)
test_dataset = build_dataset(test_df, MAX_SEQ_LEN).batch(BATCH_SIZE)

# 构建BERT分类模型

from transformers import TFBertForSequenceClassification

# 使用TF版本的中文base bert分类模型

model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-chinese',  # base中文bert
    num_labels=NUM_LABELS  # 指定输出的类别数
)

# BERT学习率一般较小, 使用Adam优化器 3e-5, 3e-6
LR = 3e-6

# BERT参数量大，拟合能力较强，在这个数据集上不需要太多迭代
EPOCHS = 5

# 同样早停等待次数也设置小一些
PATIENCE = 1

# 常用Adam优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
# 这里的标签值并不是one-hot的，所以loss需要SparseCategoricalCrossentropy
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # from_logits为True会用softmax将y_pred转化为概率，结果更稳定
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric]
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=PATIENCE,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=[callback],
    validation_data=val_dataset
)
# BERT模型保存

save_model_path = './bert/bert_classification'
model.save_pretrained(save_model_path, saved_model=True)

# 模型评估
# 结果包括loss和logits, 取出模型预测Logits

output = model.predict(test_dataset)
preds = np.argmax(output.logits, axis=-1)
preds[:10]

from sklearn.metrics import classification_report

test_label = test_df['label']
result = classification_report(test_label, preds)
print(result)
# 加载保存好的模型

save_model_path = "./bert/bert_classification"
saved_model = TFBertForSequenceClassification.from_pretrained(
    save_model_path,
    num_labels=NUM_LABELS
)
saved_model.summary()

predict_sentences = [
    "因为疫情被困家里2个月了，好压抑啊，感觉自己抑郁了！",
    "我国又一个新冠病毒疫苗获批紧急使用。",
    "我们在一起，打赢这场仗，抗击新馆疫情，我们在行动！"
]
# 调用中文bert base模型的tokenzier
predict_inputs = tokenizers(
    predict_sentences,
    padding=True,
    max_length=MAX_SEQ_LEN,
    return_tensors="tf"
)

# 直接call保存好的bert model
output = saved_model(predict_inputs)

# 取出模型预测结果的logits
predict_logits = output.logits.numpy()

# 取出分数最高的标签
predict_results = np.argmax(predict_logits, axis=1)
# 还原标签
predict_labels = [label - 1 for label in predict_results]
print(predict_labels)

# 格式化预测结果
for text, label in zip(predict_sentences, predict_labels):
    print(f'文本: {text}\n预测标签: {label}')
# 模型优化
"""
BERT是一种预训练语言模型，参数量较大，训练较慢。    
可以将BERT作为embedding层，固定其参数，只做前向运算，再接其他特征抽取层进行特征抽取。
"""
from transformers import TFBertModel

bert_model = TFBertModel.from_pretrained('bert-base-chinese')

input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name='input_ids', dtype='int32')
token_type_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name='token_type_ids', dtype='int32')
attention_masks = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name='attention_mask', dtype='int32')
embedding_layer = bert_model(input_ids, attention_masks)[0]
X = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(
        100,
        return_sequences=True,
        dropout=0.1
    )
)(embedding_layer)
X = tf.keras.layers.GlobalMaxPool1D()(X)  # 进行Max Pooling
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Dense(256, activation='relu')(X)
X = tf.keras.layers.Dropout(0.5)(X)
y = tf.keras.layers.Dense(3, activation='softmax', name='output')(X)

model = tf.keras.Model(
    inputs=[input_ids, attention_masks, token_type_ids],
    outputs=y
)
for layer in model.layers[:3]:
    layer.trainable = False

model.summary()
