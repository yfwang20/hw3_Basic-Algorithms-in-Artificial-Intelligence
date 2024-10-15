import mnist_dataloader


import tensorflow
# from tensorflow.keras import layers, models, losses, optimizers
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 定义残差块
class ResidualBlock(tensorflow.keras.layers.Layer):
    def __init__(self, units=100):
        super(ResidualBlock, self).__init__()
        self.dense1 = tensorflow.keras.layers.Dense(units, activation='relu')
        self.dense2 = tensorflow.keras.layers.Dense(units, activation=None)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return tensorflow.nn.relu(x + inputs)

# 定义神经网络模型
def create_model(input_shape=(784,)):
    model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Input(shape=input_shape),
        ResidualBlock(784),  # 使用784个节点以保持维度一致
        tensorflow.keras.layers.Dense(10, activation='softmax')  # 输出层
    ])
    return model

mnist_dataset = mnist_dataloader.read_data_sets("./MNIST_dataset/")
dataset_A, dataset_B, dataset_C = mnist_dataset.train, mnist_dataset.test, mnist_dataset.multi

train_size = dataset_A.num_examples
test_size = dataset_B.num_examples
print('Dataset size: ', '(train, test) =', (train_size, test_size))
# you can use index to get specific item (e.g. image_A[0])
images_A, images_B, images_C = dataset_A.images, dataset_B.images, dataset_C.images
labels_A, labels_B, labels_C = dataset_A.labels, dataset_B.labels, dataset_C.labels


# 模拟数据
X_train = images_A
y_train = labels_A
X_test = images_B
y_test = labels_B

# 创建模型
model = create_model()

# 编译模型
model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
              loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# 训练过程
start_time = time.time()
history = model.fit(X_train, y_train, epochs=20, batch_size=30, validation_data=(X_test, y_test), verbose=0)
end_time = time.time()

# 绘制训练和验证损失曲线
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('task5_training_test_loss.png')  # 保存图像
plt.show()

# 绘制训练和验证准确率曲线
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('task5_training_test_accuracy.png')  # 保存图像
plt.show()

# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# 预测并计算混淆矩阵
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('task5_confusion_matrix.png')  # 保存图像
plt.show()

print(f"Total training time: {end_time - start_time:.2f} seconds")