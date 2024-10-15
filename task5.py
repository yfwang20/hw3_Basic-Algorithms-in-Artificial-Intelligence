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
X_train = images_A.reshape(-1, 28, 28, 1)  # 将图像展平并归一化
X_test = images_B.reshape(-1, 28, 28, 1)
y_train = tensorflow.keras.utils.to_categorical(labels_A, 10)  # 将标签转换为one-hot编码
y_test = tensorflow.keras.utils.to_categorical(labels_B, 10)


# 定义残差块
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    if stride != 1 or x.shape[-1] != filters:
        shortcut = tensorflow.keras.layers.Conv2D(filters, 1, strides=stride, padding='same')(x)
        shortcut = tensorflow.keras.layers.BatchNormalization()(shortcut)

    x = tensorflow.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('relu')(x)
    x = tensorflow.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Add()([x, shortcut])
    x = tensorflow.keras.layers.Activation('relu')(x)
    return x

def simplified_residual_block(x, filters, kernel_size=3, stride=1):
    # 如果步长不是1或者输入通道数与输出通道数不匹配，则需要调整shortcut路径
    if stride != 1 or x.shape[-1] != filters:
        shortcut = tensorflow.keras.layers.Conv2D(filters, 1, strides=stride, padding='same')(x)
    else:
        shortcut = x

    # 主路径
    x = tensorflow.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('sigmoid')(x)

    # 残差连接
    x = tensorflow.keras.layers.Add()([x, shortcut])
    return x

# 定义训练函数
def train_model(num_nodes, num_blocks):
    inputs = tensorflow.keras.Input(shape=(28, 28, 1))
    x = tensorflow.keras.layers.Conv2D(num_nodes, 3, padding='same')(inputs)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.Activation('sigmoid')(x)

    for _ in range(num_blocks):
        x = simplified_residual_block(x, num_nodes)

    x = tensorflow.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tensorflow.keras.layers.Dense(10, activation='softmax')(x)

    model = tensorflow.keras.models.Model(inputs, outputs)

    # 编译模型
    model.compile(optimizer='adam',  # 使用 Adam 优化器
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
    for epoch in range(10):  # 可以根据需要调整epoch数量
        history = model.fit(X_train, y_train,
                            epochs=1,  # 每次只训练一个epoch
                            batch_size=30,  # 批量大小
                            validation_split=0.2,  # 使用20%的数据作为验证集
                            verbose=1)  # 不打印进度条

        # 记录训练损失和准确率
        train_loss.append(history.history['loss'][0])
        train_acc.append(history.history['accuracy'][0])

        # 评估测试集上的性能
        test_results = model.evaluate(X_test, y_test, verbose=0)
        test_loss.append(test_results[0])
        test_acc.append(test_results[1])

    # 计算运行时间
    end_time = time.time()
    print(f"Training with {num_nodes} nodes and {num_blocks} blocks completed in {end_time - start_time:.2f} seconds.")

    return train_loss, train_acc, test_loss, test_acc, model

# 隐含层节点数列表
num_nodes = 50  # 你可以根据需要调整节点数
# 隐含层层数列表
num_blocks_list = [1]  # 你可以根据需要调整层数

# 存储结果
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
models_list = []

# 训练不同层数的模型
for num_blocks in num_blocks_list:
    train_loss, train_acc, test_loss, test_acc, model = train_model(num_nodes, num_blocks)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    models_list.append(model)

# 绘制训练错误率曲线和测试错误率曲线
plt.figure(figsize=(12, 8))

for i, num_blocks in enumerate(num_blocks_list):
    plt.plot(train_losses[i], label=f'{num_blocks} Blocks (Train Loss)')
    plt.plot(test_losses[i], label=f'{num_blocks} Blocks (Test Loss)')

plt.title('Training and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('task5_training_test_loss.png')  # 保存图像
plt.show()

# 绘制训练准确率曲线和测试准确率曲线
plt.figure(figsize=(12, 8))

for i, num_blocks in enumerate(num_blocks_list):
    plt.plot(train_accuracies[i], label=f'{num_blocks} Blocks (Train Accuracy)')
    plt.plot(test_accuracies[i], label=f'{num_blocks} Blocks (Test Accuracy)')

plt.title('Training and Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('task5_training_test_accuracy.png')  # 保存图像
plt.show()



# 生成混淆矩阵
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# 在程序最末尾绘制所有模型的混淆矩阵
for i, num_blocks in enumerate(num_blocks_list):
    model = models_list[i]
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, f'Confusion Matrix for {num_nodes} nodes and {num_blocks} Blocks')

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
for i, num_blocks in enumerate(num_blocks_list):
    model = models_list[i]
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, f'Confusion Matrix for {num_nodes} nodes and {num_blocks} Blocks', f'task5_confusion_matrix_{num_nodes}_nodes_and_{num_blocks}_Blocks.png')