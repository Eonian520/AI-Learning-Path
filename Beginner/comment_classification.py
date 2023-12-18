"""
该项目将使用IMDB数据库, 将识别评论文本为积极或消极两个分类, IMDB数据库说明
    1. 标签值为0或1, 其中0代表消极评论, 1代表积极评论
    2. 评论文本被转换为整数值, 每个整数代表词典中的一个单词, 例如: [1, 14, 22, 16, 43]

通过该项目, 应该学习到:
    1. 如何对于不同长度的输入, 进行神经网络学习

参考资料: https://tensorflow.google.cn/tutorials/keras/text_classification?hl=zh-cn
"""

import tensorflow as tf
import keras


def main():
    # 获取数据集, 值得注意的是不同评论的长度是不同的
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)  # 保留训练数据中最常出现的10000个单词

    # 对数据作特殊处理: 构建单词表, 并构建评论自动转换函数
    word_index_map = imdb.get_word_index()  # 建立单词到整数索引的字典
    word_index = {k: (v + 3) for k, v in word_index_map.items()}  # 保留前三个索引序号用于特殊处理
    word_index["<PAD>"], word_index["<START>"], word_index["<UNK>"], word_index["<UNUSED>"] = 0, 1, 2, 4

    index_word_map = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):  # 自动将整数索引转化为评论
        return ' '.join([index_word_map.get(i, '?') for i in text])

    # 将输入转换为张量, 将所有数据转换为 max_length * num_reviews 长度的张量以统一输入数据长度
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<PAD>"],  # 填充值为 word_index["<PAD>"]
                                                            padding='post',     # 在末尾填充
                                                            maxlen=256)         # 张量长度为256

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)

    # 定义损失函数, 构建并编译模型
    vocab_size = 10000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()

    # 训练模型

    # 测试模型性能


if __name__ == "__main__":
    main()
