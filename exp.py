from __future__ import print_function, division

from keras.layers import Input, Dense, Activation
from keras.layers.merge import Maximum, Concatenate
from keras.models import Model
from keras.optimizers import Adam

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

class MalGAN():
    def __init__(self):
        self.apifeature_dims = 128
        self.z_dims = 20   #could try 20,malware想欺骗分类器就得增加api调用
                           #z噪声设置的越大，可能增加的api数量就会越多

        self.hide_layers = 256
        self.generator_layers = [self.apifeature_dims+self.z_dims, self.hide_layers, self.apifeature_dims]#128+20,256,128
        self.substitute_detector_layers = [self.apifeature_dims, self.hide_layers, 1]#128,256
        self.blackbox = 'MLP'
        optimizer = Adam(lr=0.001)

        # Build and Train blackbox_detector
        self.blackbox_detector = self.build_blackbox_detector()#黑盒

        # Build and compile the substitute_detector
        self.substitute_detector = self.build_substitute_detector()#检测器
        self.substitute_detector.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])#binary_crossentropy亦称作对数损失
        #损失函数
        # Build the generator
        self.generator = self.build_generator()

        # The generator takes malware and noise as input and generates adversarial malware examples
        example = Input(shape=(self.apifeature_dims,))#(?,128)
        #print("example:",example.shape)
        noise = Input(shape=(self.z_dims,))#(?,20)
        input = [example, noise]
        malware_examples = self.generator(input)

        # For the combined model we will only train the generator 为True时才根据训练结果自动更新参数
        self.substitute_detector.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.substitute_detector(malware_examples)#样本有效性（128,1）

        # The combined model  (stacked generator and substitute_detector)
        # Trains the generator to fool the discriminator
        self.combined = Model(input, validity)#（输入，输出）
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_blackbox_detector(self):

        if self.blackbox is 'MLP':
            blackbox_detector = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                                              solver='sgd', verbose=0, tol=1e-4, random_state=1,
                                              learning_rate_init=.1)
        return blackbox_detector

    def build_generator(self):

        example = Input(shape=(self.apifeature_dims,))
        noise = Input(shape=(self.z_dims,))
        x = Concatenate(axis=1)([example, noise])#(128,148)
        for dim in self.generator_layers[1:]:#[256,128]
            x = Dense(dim)(x)
            x = Activation(activation='sigmoid')(x) #全连接层 激活函数 output 隐藏层，输出层
        x = Maximum()([example, x])#example|x#(?,128)
        generator = Model([example, noise], x, name='generator') #（输入，输出）
        generator.summary()
        return generator

    def build_substitute_detector(self):

        input = Input(shape=(self.substitute_detector_layers[0],))
        x = input
        for dim in self.substitute_detector_layers[1:]:
            x = Dense(dim)(x)
            x = Activation(activation='sigmoid')(x) 
        substitute_detector = Model(input, x, name='substitute_detector')#（输入，输出）
        substitute_detector.summary()
        return substitute_detector

    def load_data(self, filename):

        data = np.load(filename)
        xmal, ymal, xben, yben = data['xmal'], data['ymal'], data['xben'], data['yben']
        return (xmal, ymal), (xben, yben)

    def train(self, epochs, batch_size=32):

        # Load the dataset
        (xmal, ymal), (xben, yben) = self.load_data('E://data/data.npz')
        xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(xmal, ymal, test_size=0.20)#20%测试 80%训练1368
        xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(xben, yben, test_size=0.20)#441
        #print("xtrain_mal",xtrain_mal.shape) #（1094,128）
        #print(np.concatenate([xmal, xben]).shape)#(1809,128)
        #print(np.concatenate([ymal, yben]).shape)#(1809,)
        # Train blackbox_detctor
        self.blackbox_detector.fit(np.concatenate([xmal, xben]),
                                   np.concatenate([ymal, yben]))

        ytrain_ben_blackbox = self.blackbox_detector.predict(xtrain_ben)#（352，）
        Original_Train_TRR = self.blackbox_detector.score(xtrain_mal, ytrain_mal)#原始训练样本检测率
        Original_Test_TRR = self.blackbox_detector.score(xtest_mal, ytest_mal)#原始测试样本检测率
        Train_TRR, Test_TRR = [], []

        for epoch in range(epochs):#1000

            for step in range(1):#range(xtrain_mal.shape[0] // batch_size):(1094//128=8)
                # ---------------------
                #  Train substitute_detector
                # ---------------------

                # Select a random batch of malware examples
                idx = np.random.randint(0, xtrain_mal.shape[0], batch_size) #[0,1094) 共128个
                xmal_batch = xtrain_mal[idx]#(128,128)
                noise = np.random.uniform(0, 1, (batch_size, self.z_dims))#(128,20)
                idx = np.random.randint(0, xmal_batch.shape[0], batch_size)#[0,128) 共128个
                xben_batch = xtrain_ben[idx]#（128,128）
                #print(xben_batch.shape)
                yben_batch = ytrain_ben_blackbox[idx]#（128，）
                #print(yben_batch.shape)
                # Generate a batch of new malware examples
                gen_examples = self.generator.predict([xmal_batch,noise])##(128,128)
                ymal_batch = self.blackbox_detector.predict(np.ones(gen_examples.shape)*(gen_examples > 0.5))#（128,）

                # Train the substitute_detector
                d_loss_real = self.substitute_detector.train_on_batch(gen_examples, ymal_batch)#对抗样本损失
                d_loss_fake = self.substitute_detector.train_on_batch(xben_batch, yben_batch)#良性样本损失
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                idx = np.random.randint(0, xtrain_mal.shape[0], batch_size)#[0,1094] 共128
                xmal_batch = xtrain_mal[idx]#(128,128)
                noise = np.random.uniform(0, 1, (batch_size, self.z_dims))#(128,20)

                # Train the generator
                g_loss = self.combined.train_on_batch([xmal_batch,noise], np.zeros((batch_size, 1)))

            # Compute Train TRR
            noise = np.random.uniform(0, 1, (xtrain_mal.shape[0], self.z_dims))#（1094,20）
            gen_examples = self.generator.predict([xtrain_mal, noise])#（（1094,128），（1094,20）） ##（1094,128）
            TRR = self.blackbox_detector.score(np.ones(gen_examples.shape) * (gen_examples > 0.5), ytrain_mal)#对抗训练样本检测率
            Train_TRR.append(TRR)

            # Compute Test TRR
            noise = np.random.uniform(0, 1, (xtest_mal.shape[0], self.z_dims))
            gen_examples = self.generator.predict([xtest_mal, noise])
            TRR = self.blackbox_detector.score(np.ones(gen_examples.shape) * (gen_examples > 0.5), ytest_mal)
            Test_TRR.append(TRR)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        print('Original_Train_TRR: {0}, Adver_Train_TRR: {1}'.format(Original_Train_TRR, Train_TRR[-1]))
        print('Original_Test_TRR: {0}, Adver_Test_TRR: {1}'.format(Original_Test_TRR, Test_TRR[-1]))

        # Plot TRR
        plt.figure()
        plt.plot(range(epochs), Train_TRR, c='r', label='Training Set', linewidth=2)#训练集检测率
        plt.plot(range(epochs), Test_TRR, c='g', linestyle='--', label='Validation Set', linewidth=2)#测试集检测率
        plt.xlabel("Epoch")
        plt.ylabel("TRR")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    malgan = MalGAN()
    malgan.train(epochs=1000, batch_size=128)
