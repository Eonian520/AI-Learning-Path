"""
利用预构建的MNIST数据集进行模型训练, 应该学习到:
    1. 了解模型训练步骤
    2. 了解模型配置方法

参考资料: https://tensorflow.google.cn/tutorials/quickstart/beginner?hl=zh-cn
"""

import tensorflow as tf
import keras


def main():
    # 步骤1: 获取数据集
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 使用数据集前先做张量转换处理
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 步骤2: 构建训练模型
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),     # Flatten层将输入从多维压缩到一维中, 方便矩阵运算, 并不会影响批处理维度
        keras.layers.Dense(128, activation='relu'),     # 输入节点为128, 激活函数为rule (利用激活函数将线性模型转变为非线性模型)
        keras.layers.Dropout(0.2),      # 设置频率rate随机对输入置零以防止过拟合, 其他输入同比例大 1/(1 - rate) 倍
        keras.layers.Dense(10)          # Dense层是每个输出点都与输入点加权相关, 因此叫稠密, 是输入到输出的映射
    ])

    # 步骤3: 定义损失函数
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # 定义损失函数

    # predictions = model(x_train[:1]).numpy()  # 该模型返回对每个标签进行预测的对数分数, 这里只取第一个数据运行模型
    # print(f"prediction probability before train is: {tf.nn.softmax(predictions).numpy()}")  # 利用softmax函数将对数分数转化为概率
    # print(f"loss score before train is: {loss_fn(y_train[:1], predictions).numpy()}")   # 未经过训练的损失大约是 log(1/10) ~= 2.3

    # 步骤4: 配置并编译模型
    model.compile(optimizer='adam',     # 化器类型为adam
                  loss=loss_fn,         # 设置损失函数
                  metrics=['accuracy']  # 使用精准度作为模型的性能衡量
                  )

    # 步骤5: 开始训练模型
    model.fit(x_train, y_train, epochs=5)  # 训练模型, 迭代5个周期

    # 步骤6: 模型性能测试
    model.evaluate(x_test, y_test, verbose=2)  # 利用验证集或测试集测试性能, verbose表示输出日志的语法模式


if __name__ == "__main__":
    main()
