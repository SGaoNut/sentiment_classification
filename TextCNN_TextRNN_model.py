# -*- coding: utf-8 -*-

"""
@author: shan
@software: PyCharm
@file: TextCNN_TextRNN_model.py
@time: 2021/8/29 11:09 上午
"""
import re

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

random_seed = 100

import warnings

warnings.filterwarnings('ignore')


# 数据预处理与数据集划分

def preprocess_text(words):
    # 转小写
    words = words.lower()
    # 去除标点，保留中英文字符
    words = re.sub('[^a-z^0-9^\u4e00-\u9fa5]', '', words)
    # 将所有数字都转为0
    words = re.sub('[0-9]]', '0', words)
    words = ' '.join(words)
    # 合并连续的空格
    words = re.sub('\s{2,}', ' ', words)
    return words.strip()


raw_train_df = pd.read_csv('./processed_data/train.csv')
raw_train_df['text'] = raw_train_df['text'].apply(preprocess_text)
raw_train_df = raw_train_df[raw_train_df['text'] != '']
# 过滤掉处理后文本为空的数据
print(raw_train_df[['text', 'label']].head())

test_df = pd.read_csv('./processed_data/test.csv')
test_df['text'] = test_df['text'].apply(preprocess_text)
test_df = test_df[test_df['text'] != '']
print(test_df[['text', 'label']].head())

# 划分训练集与验证集

train_df, val_df = train_test_split(
    raw_train_df,
    test_size=5000,
    random_state=1234
)
print('训练集大小:{}'.format(len(train_df)))
print('验证集大小:{}'.format(len(val_df)))

# 保存划分好的数据集
train_df.to_csv('./dataset/train.csv', index=False)
val_df.to_csv('./dataset/val.csv', index=False)
test_df.to_csv('./dataset/test.csv', index=False)

"""
构建TF数据集   
TensorFlow全新的数据读取方式：
Dataset API入门教程：https://zhuanlan.zhihu.com/p/30751039
"""

# 读取原始数据集
train_text, train_label = train_df['text'], train_df['label']
val_text, val_label = val_df['text'], val_df['label']
test_text, test_label = test_df['text'], test_df['label']
print('训练集大小：{}'.format(len(train_label)))
print('验证集大小：{}'.format(len(val_label)))
print('测试集大小：{}'.format(len(test_label)))

"""
# Tokenizer
Tokenizer是一个分词器，用于文本预处理，序列化，向量化等。  
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer 
默认情况下，将删除所有标点符号，从而将文本转换为以空格分隔的单词序列（单词可能包含’字符，如I’am）,
然后将这些序列分为标记列表，并将它们编入索引或向量化。注意0是保留索引，不会分配给任何单词。
"""

tokenizers = tf.keras.preprocessing.text.Tokenizer(
    # 根据单词频率排序，保留前num_words个单词，即仅保留最常见的num_words-1个单词
    num_words=None,
    # 一个用于过滤的正则表达式的字符串，这个过滤器作用在每个元素上，
    # 默认过滤除‘`’字符外的所有标点符号，制表符和换行符
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    # 标记是否将文本转换为小写
    lower=True,
    # 分词分隔符，前面已经处理成空格分割了
    split=' ',
    # 是否进行字符级别的粉刺，前面已经分好了
    char_level=False,
    # 指定out of vocabulary词，将被添加到word_index中，
    # 并用于在text_to_sequence调用期间替换词汇外的单词，即用来补充原文本中没有的词。
    oov_token=None
)

# 用训练数据的文本训练tokenizer
tokenizers.fit_on_texts(train_text)

text = '写在年末冬初孩子流感的第五天，我们仍然没有忘记热情拥抱这2020年的第一天。'
# 将文本数据转化为对应id序列
tokens = tokenizers.texts_to_sequences(text)
print(tokens)
# 保存tokenizer用于预测
tokenizers_save_path = './model/tokenizer.pkl'
joblib.dump(tokenizers, tokenizers_save_path)

"""
### tf dataset
官方API： https://www.tensorflow.org/api_docs/python/tf/data/Dataset
其他：https://www.huaweicloud.com/articles/5fcbf05e61828f8e6eeced07ccedc3c6.html

- from_tensor_slices: 用于创建dataset,其元素是给定张量的切片的元素。
- shuffle: 随机打乱此数据集的样本。从data数据集中按顺序抽取buffer_size个样本放在buffer中，然后打乱buffer中的样本。
buffer中样本个数不足buffer_size，继续从data数据集中安顺序填充至buffer_size，此时会再次打乱。
对于完美的打乱，缓冲区大小需要大于或等于数据集的大小。
- batch: 将数据集的样本构造成批数据。
- prefetch: 创建一个从该数据集中预先读取元素的数据集。大多数数据集输入管道应该以调用预取结束。这允许在处理当前元素时准备后面的元素。这通常会提高延迟和吞吐量，但代价是使用额外的内存来存储预取元素。

pad_sequences是做补齐序列作用的
"""

# 类别数量
NUM_LABEL = 3
# 构造数据集的bacth
BATCH_SIZE = 64
# 最长序列长度，经前面分析，这里取240
MAX_LEN = 240
# bacth读取数据的缓存大小
BUFFER_SIZE = tf.constant(len(train_text), dtype=tf.int64)


def build_dataset(words, label, is_train=False):
    sequence = tokenizers.texts_to_sequences(words)  # 将前面的tokenzier用于当前文本进行id序列化
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=MAX_LEN)  # 把所有序列补齐到最长序列长度，padding方式为在后面补齐
    label_tensor = tf.convert_to_tensor(
        tf.one_hot(label, NUM_LABEL),
        dtype=tf.float32
    )  # 将标签转化为tf的one-hot标签，即 1 -> [0, 1, 0]

    dataset = tf.data.Dataset.from_tensor_slices(
        (padded_sequence, label_tensor)
    )  # 将序列化的文本数据与标签在一起制作tf.dataset

    # 训练阶段
    if is_train:
        dataset = dataset.shuffle(BUFFER_SIZE)  # 训练阶段将数据打乱
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)  # 按照batch_size构造数据集，去掉不能组成batch的多余数据
        dataset = dataset.prefetch(BUFFER_SIZE)  # 训练时预先将所有的训练数据加载到内存中
    else:
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
        dataset = dataset.prefetch(BATCH_SIZE)
    return dataset


# 分别构造训练、验证、测试tf数据集
train_dataset = build_dataset(train_text, train_label, is_train=True)
val_dataset = build_dataset(val_text, val_label, is_train=False)
test_dataset = build_dataset(test_text, test_label, is_train=False)

for example, label in train_dataset.take(1):
    print('texts: ', example.numpy()[:3])
    print()
    print('labels: ', label.numpy()[:3])

# 模型构建与训练

VOCAB_SIZE = len(tokenizers.index_word) + 1  # 词典大小，留一个unk字符
EMBEDDING_DIM = 100  # 词向量大小

# from gensim.models.keyedvectors import KeyedVectors


# def get_embeddings(pretrain_vec_path):
#     word_vectors = KeyedVectors.load_word2vec_format(
#         pretrain_vec_path,
#         binary=False
#     )  # 加载训练好的词向量为KeyedVectors
#
#     word_vocab = set(word_vectors.key_to_index.keys())  # 去除预训练词向量的词汇表
#     token_embeddings = np.random.uniform(
#         -0.2,
#         0.2,
#         size=(VOCAB_SIZE, EMBEDDING_DIM)
#     )
#     for i in range(1, VOCAB_SIZE):  # 从1开始，0是tokenziers预留的索引
#         word = tokenizers.index_word[i]  # 取出tokenziers索引对应的词
#         if word in word_vocab:  # 如果词出现在预训练的词中
#             token_embeddings[i, :] = word_vectors.get_vector(word)  # 就将预训练的此项俩姑娘赋值给它，替换随机初始化
#     return token_embeddings
#
#
# pretrained_vec_path = './word2vec/sg_ns_100.txt'
# # 加载预训练词向量
# embeddings = get_embeddings(pretrained_vec_path)
# embeddings[:2]

# 模型定义
FILTERS = [2, 3, 5]  # 3种卷积核的filter size
NUM_FILTERS = 128  # 卷积核的大小
DENSE_DIM = 256  # 全连接层大小
CLASS_NUM = 3  # 类别数
DROPOUT_RATE = 0.5  # dropout比例


# 定义一个TextCNN的模型

def build_text_cnn_model():
    inputs = tf.keras.Input(shape=(None,), name='input_data')  # 模型输入
    embed = tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,  # 词表大小
        output_dim=EMBEDDING_DIM,  # 词向量维度
        # embeddings_initializer=tf.keras.initializers.Constant(embeddings),  # 加载预训练的词向量
        trainable=True,  # 词向量是否训练
        mask_zero=True  # 标记是否用0进行mask标记序列补齐到固定长度
    )(inputs)
    # 在embedding层后接dropout
    embed = tf.keras.layers.Dropout(DROPOUT_RATE)(embed)

    pool_outputs = []
    for filter_size in FILTERS:
        conv = tf.keras.layers.Conv1D(
            filters=NUM_FILTERS,  # 卷积核个数
            kernel_size=filter_size,  # 卷积核尺寸
            padding='same',  # same padding
            activation='relu',  # 激活函数
            data_format='channels_last',  # 数据合适，最后一维是通道
            use_bias=True  # 标记是否使用bias
        )(embed)  # 分别对文本进行应用不同filter size的卷积
        max_pool = tf.keras.layers.GlobalMaxPool1D(
            data_format='channels_last'
        )(conv)
        pool_outputs.append(max_pool)  # 将对应filter size大小卷积池化后的结果存到list

    outputs = tf.keras.layers.concatenate(pool_outputs, axis=-1)  # 将不同filter size 卷积池化后的结果进行拼接
    outputs = tf.keras.layers.Dense(DENSE_DIM, activation='relu')(outputs)  # dense层
    outputs = tf.keras.layers.Dropout(DROPOUT_RATE)(outputs)  # Dropout
    outputs = tf.keras.layers.Dense(CLASS_NUM, activation='softmax')(outputs)  # 最终的分类层
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


text_cnn_model = build_text_cnn_model()
text_cnn_model.summary()

LR = 3e-4  # 学习率
EPOCHS = 1
PATIENCE = 5

loss = tf.keras.losses.CategoricalCrossentropy()
optimizers = tf.keras.optimizers.Adam(learning_rate=LR)

text_cnn_model.compile(
    loss=loss,
    optimizer=optimizers,
    metrics=['accuracy']
)
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  # 通过验证集的accuracy做监控指标，几个epoch该指标不再上升，则早停
    patience=PATIENCE,
    restore_best_weights=True  # 标记是否还原在监控指标中渠道最好结果的模型权重
)  # 增加早停

# 样本分布不均衡的解决：通过对不同标签的损失进行加权
# 这里标签1的数据最多，降低其权重
class_weight = {0: 0.4, 1: 0.2, 2: 0.4}
# 模型训练
history = text_cnn_model.fit(
    train_dataset,  # 输入训练数据集
    epochs=EPOCHS,
    callbacks=[callback],  # early stopping
    validation_data=val_dataset,
    class_weight=class_weight
)
# 在测试集上进行评估
test_loss, test_acc = text_cnn_model.evaluate(test_dataset)

print('Test Loss', test_loss)
print('Test Accuracy', test_acc)

import matplotlib.pyplot as plt


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])


# 将训练过程进行可视化
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)

# 保存训练好的模型
model_save_path = './model/text_cnn'

text_cnn_model.save(model_save_path)

# 对测试集进行预测，得到每个测试样本的在各个标签的分数
predictions = text_cnn_model.predict(test_dataset)
predictions[:10]

# 取出每个测试样本分数最大的标签作为预测标签
preds = np.argmax(predictions, axis=-1)
preds[:10]

from sklearn.metrics import classification_report

# 查看分类结果
result = classification_report(test_label, preds)
print(result)

"""
4. 模型预测

训练好的模型如何进行在线预测
1. 加载保存好的模型
2. 对预测文本进行预处理、tokenizer以及padding
3. 输入模型进行预测
4. 展示预测结果
"""

# 加载训练好的Tokenzier
tokenizer_save_path = './model/tokenizer.pkl'
tokenizer = joblib.load(tokenizer_save_path)

# 加载TextCNN模型
model_save_path = './model/text_cnn'
text_cnn_model = tf.keras.models.load_model(model_save_path)
text_cnn_model.summary()

predict_sentences = [
    "因为疫情被困家里2个月了，好压抑啊，感觉自己抑郁了！",
    "我国又一个新冠病毒疫苗获批紧急使用。",
    "我们在一起，打赢这场仗，抗击新馆疫情，我们在行动！"]

# 预处理文本
predict_texts = [preprocess_text(text) for text in predict_sentences]
# 进行tokenizer
predict_sequences = tokenizer.texts_to_sequences(predict_texts)
# 进行padding
sequence_padded = pad_sequences(
    predict_sequences,
    padding='post',
    maxlen=MAX_LEN
)
# 得到预测的Logits
predict_logits = text_cnn_model.predict(sequence_padded)
predict_logits

# 取出分数最高的标签
predict_results = np.argmax(predict_logits, axis=1)
# 还原标签
predict_labels = [label - 1 for label in predict_results]
predict_labels

# 展示预测结果
for text, label in zip(predict_sentences, predict_labels):
    print(f'Text: {text}\nPredict: {label}')

# 模型优化
"""
TextRNN
RNN/LSTM 相对CNN更能捕捉序列信息，在短文本分类领域更适合
"""

LSTM_DIM = 256
DENSE_DIM = 128

# 通过Sequential方式定义TextRNN model
# 其他层都与TextCNN一致
# 将特征抽取层替换为BiLSTM

text_rnn_model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            # embeddings_initializer=tf.keras.initializers.Constant(embeddings),
            trainable=True,
            mask_zero=True
        ),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        # # 若两层BiLSTM, 第一层应return_sequences，然后继续接一层
        # tf.keras.layers.Bidirectional(
        #     tf.keras.layers.LSTM(LSTM_DIM * 2,  return_sequences=True)
        # ),
        # 加一层双向LSTM进行特征抽取
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(LSTM_DIM)
        ),
        tf.keras.layers.Dense(DENSE_DIM, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(CLASS_NUM, activation='softmax')
    ]
)

text_rnn_model.summary()
"""
上采样与下采样
解决数据不平衡问题，可对数据比较多的标签进行下采样，对数据较少的标签进行上采样，达到数据集标签分布的平衡   
下面是如何通过Tf dataset的方式构造平衡的数据集   
在前面的分析中各标签数据分布为: [17.96, 56.83, 25.21]   
这里构造一个相对平衡的数据集，设置各标签数据分布为 [0.3, 0.4, 0.3]
"""

labels = [0, 1, 2]  # 对[0, 1, 2]标签的数据不同的采样权重，强行定义采样
balance_weights = [0.3, 0.4, 0.3]
dataset_each_label = []

for current_label in labels:
    current_label_dataset = train_dataset.unbatch(). \
        filter(lambda example, label: tf.argmax(label) == current_label).repeat()
    dataset_each_label.append(current_label_dataset)

# 通过相对平衡的采样权重，构造相对平衡的数据集
balances_train_dataset = tf.data.experimental.sample_from_datasets(
    dataset_each_label, weights=balance_weights
).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(BUFFER_SIZE)

for example, label in balances_train_dataset.take(1):
    print('texts: ', example.numpy()[:3])
    print()
    print('labels: ', label.numpy()[:3])

text_cnn_model_new = build_text_cnn_model()
text_cnn_model_new.compile(
    loss=loss,
    optimizer=optimizers,
    metrics=['accuracy']
)
# 由于采样后构造的数据集是无限的，所以模型训练的方式需要做一些改变
# 需要指定每个epoch跑多少个step，因为每个step随机从数据集中采样相对平衡的数据集
history = text_cnn_model_new.fit(
    balances_train_dataset,  # 指定平衡后的训练数据集
    epochs=EPOCHS,
    callbacks=[callback],
    validation_data=val_dataset,
    class_weight=class_weight,
    steps_per_epoch=20000,  # 指定每个epoch跑多少step
)

# 查看新数据集的模型结果
predictions = text_cnn_model.predict(test_dataset)
preds = np.argmax(predictions, axis=-1)
result = classification_report(test_label, preds)
print(result)
