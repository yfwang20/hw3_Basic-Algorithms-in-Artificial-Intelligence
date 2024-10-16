import os, sys
import numpy as np

import mnist_dataloader

import tensorflow
# from tensorflow.keras import layers, models
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical
# layers = 'tensorflow.keras.layers'
# models = 'tensorflow.keras.models'
# to_categorical = 'tensorflow.keras.utils.to_categorical'
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns



mnist_dataset = mnist_dataloader.read_data_sets("./MNIST_dataset/")
dataset_A, dataset_B, dataset_C = mnist_dataset.train, mnist_dataset.test, mnist_dataset.multi

train_size = dataset_A.num_examples
test_size = dataset_B.num_examples
print('Dataset size: ', '(train, test) =', (train_size, test_size))
# you can use index to get specific item (e.g. image_A[0])
images_A, images_B, images_C = dataset_A.images, dataset_B.images, dataset_C.images
labels_A, labels_B, labels_C = dataset_A.labels, dataset_B.labels, dataset_C.labels

# 数据预处理
X_train = images_A  # 将图像展平并归一化
X_test = images_B
y_train = tensorflow.keras.utils.to_categorical(labels_A, 10)  # 将标签转换为one-hot编码
y_test = tensorflow.keras.utils.to_categorical(labels_B, 10)



# 定义训练函数
def train_model(hidden_layers):
    # 构建模型
    model = tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)))  # 输入层到第一个隐含层
    for _ in range(hidden_layers - 1):  # 添加剩余的隐含层
        model.add(tensorflow.keras.layers.Dense(100, activation='sigmoid'))
    model.add(tensorflow.keras.layers.Dense(10, activation='softmax'))  # 输出层

    # 编译模型
    model.compile(optimizer='adam',  # 使用随机梯度下降作为优化器
                  loss='categorical_crossentropy',  # 多分类交叉熵损失函数
                  metrics=['accuracy'])

    # 记录开始时间
    start_time = time.time()

    # 存储训练和测试的损失和准确率
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # 训练模型
    for epoch in range(20):  # 可以根据需要调整epoch数量
        history = model.fit(X_train, y_train,
                            epochs=1,  # 每次只训练一个epoch
                            batch_size=20,  # 批量大小
                            verbose=0)  # 不打印进度条

        # 记录训练损失和准确率
        train_loss.append(history.history['loss'][0])
        train_acc.append(history.history['accuracy'][0])

        # 评估测试集上的性能
        test_results = model.evaluate(X_test, y_test, verbose=0)
        test_loss.append(test_results[0])
        test_acc.append(test_results[1])

    # 计算运行时间
    end_time = time.time()
    print(f"Training with {hidden_layers} hidden layers completed in {end_time - start_time:.2f} seconds.")

    # 评估模型
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test accuracy with {hidden_layers} hidden layers: {test_results[1]:.4f}')

    return train_loss, train_acc, test_loss, test_acc, model

# 隐含层层数列表
hidden_layer_counts = [1, 3, 5]

# 存储结果
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
models_list = []

# 训练不同层数的模型
for layers_count in hidden_layer_counts:
    train_loss, train_acc, test_loss, test_acc, model = train_model(layers_count)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    models_list.append(model)

# 绘制训练错误率曲线和测试错误率曲线
plt.figure(figsize=(12, 8))

for i, layers_count in enumerate(hidden_layer_counts):
    plt.plot(train_losses[i], label=f'{layers_count} Hidden Layers (Train Loss)')
    plt.plot(test_losses[i], label=f'{layers_count} Hidden Layers (Test Loss)')

plt.title('Training and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('task2_training_test_loss.png')  # 保存图像
plt.show()

# 绘制训练准确率曲线和测试准确率曲线
plt.figure(figsize=(12, 8))

for i, layers_count in enumerate(hidden_layer_counts):
    plt.plot(train_accuracies[i], label=f'{layers_count} Hidden Layers (Train Accuracy)')
    plt.plot(test_accuracies[i], label=f'{layers_count} Hidden Layers (Test Accuracy)')

plt.title('Training and Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('task2_training_test_accuracy.png')  # 保存图像
plt.show()

# 生成混淆矩阵
def plot_confusion_matrix(y_true, y_pred, title, save_filename=None):
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    if save_filename:
        plt.savefig(save_filename)  # 保存图像
    plt.show()

# 在程序最末尾绘制所有模型的混淆矩阵
for i, layers_count in enumerate(hidden_layer_counts):
    model = models_list[i]
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, f'Confusion Matrix for {layers_count} Hidden Layers', f'task2_confusion_matrix_{layers_count}_layers.png')