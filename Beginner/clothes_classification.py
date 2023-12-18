"""
利用Fashion MNIST数据集训练网络神经模型, 应该学习到:
    1. 对预测分类和真实分类的分析处理
    2. 熟练神经网络构建

参考资料:  https://tensorflow.google.cn/tutorials/keras/classification?hl=zh-cn
"""
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Classificator:
    def __init__(self):
        self.x_train = ...
        self.y_train = ...
        self.x_test = ...
        self.y_test = ...
        self.class_names = ...
        self.model = ...

    def __repr__(self):
        return "Clothes Classificator"

    def run(self) -> None:
        """

        :return:
        """
        # 导入训练数据集
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.fashion_mnist.load_data()
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # 训练集的标签名
        self.x_train, self.y_train = self.x_train / 255.0, self.y_train / 255.0  # 将像素值从0~255缩小到0~1

        # 构建神经网络模型, 先构建, 再编译
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10)
        ])

        self.model.compile(optimizer='adam',  # 优化器 - 决定模型如何根据其看到的数据和自身的损失函数进行更新
                           loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 损失函数 - 测量模型在训练期间的准确程度。
                           metrics=['accuracy'])  # 指标 - 用于监控训练和测试步骤。以下示例使用了准确率，即被正确分类的图像的比率。

        # 模型训练
        self.model.fit(self.x_train, self.y_train, epochs=10)

        # 模型评估
        test_loss, test_acc = self.model.evaluate(self.x_train, self.y_train, verbose=2)
        print('\nTest accuracy:', test_acc)

    def show_train_sample(self) -> None:
        """

        :return:
        """
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.x_train[i], cmap=plt.cm.binary)
            plt.xlabel(self.class_names[self.y_train[i]])
        plt.show()

    def predict_image(self, index: int = 1) -> None:
        """

        :param index:
        :return:
        """
        probability_model = keras.Sequential([self.model, keras.layers.Softmax()])  # 模型附加softmax层, 使得logit输出转化为概率
        prediction = probability_model.predict(self.x_test[index])  # 预测测试集中的第n个图像
        print(f"this image is predicted as: {self.class_names[np.argmax(prediction)]}")  # 输出预测结果

        # 预测可视化
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        self._plot_image(self.x_test[index], self.y_test[index], prediction)
        plt.subplot(1, 2, 2)
        self._plot_value_array(prediction, self.y_test[index])
        plt.show()

    def _plot_image(self, image, true_label, predict_array) -> None:
        """

        :param image:
        :param true_label:
        :param predict_array:
        :return:
        """
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap=plt.cm.binary)

        # 获取预测分类, 真正分类, 预测可能性
        predicted_class = self.class_names[np.argmax(predict_array)]
        true_class = self.class_names[true_label]
        predict_score = 100 * np.max(predict_array)
        color = 'blue' if predicted_class == true_class else 'red'

        # 图片横坐标标记
        x_label = f"{predicted_class} ({true_class}) {predict_score:2.0f}%"
        plt.xlabel(x_label, color=color)

    @staticmethod
    def _plot_value_array(predict_array, true_label) -> None:
        """

        :param predict_array:
        :param true_label:
        :return:
        """
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        figure = plt.bar(range(10), predict_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predict_array)
        figure[predicted_label].set_color('red')
        figure[true_label].set_color('blue')


if __name__ == "__main__":
    classificator = Classificator()
    classificator.run()
