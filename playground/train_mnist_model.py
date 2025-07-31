import tensorflow as tf
from tensorflow import keras
import os

def train_and_save_mnist_model():
    """训练并保存MNIST模型"""
    print("开始训练MNIST模型...")
    
    # 加载MNIST数据
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # 数据预处理
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 创建模型
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # 训练模型
    print("正在训练模型...")
    model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=1)
    
    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试准确率: {test_acc:.4f}")
    
    # 保存模型
    model.save('models/mnist_model.h5')
    print("模型已保存为 mnist_model.h5")
    
    return model

if __name__ == "__main__":
    train_and_save_mnist_model()
