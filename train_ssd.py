# import pickle
# from utils.detection_generate import Generator
# from utils.ssd_utils import BBoxUtility
# from nets.ssd_net import SSD300
# from utils.ssd_losses import MultiboxLoss
# from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
# import keras
# import tensorflow as tf
#
# class SSDTrain(object):
#     def __init__(self, num_classes=9, input_shape=(300, 300, 3), epochs=30):
#         """初始化网络指定一些参数，训练数据类别，图片需要指定模型输入大小，迭代次数"""
#         self.num_classes = num_classes
#         self.batch_size = 32
#         self.input_shape = input_shape
#         self.epochs = epochs
#
#         # 指定训练和读取数据的相关参数
#         self.gt_path = "./datasets/commodity_gt.pkl"
#         self.image_path = "./datasets/commodity/JPEGImages/"
#
#         prior = pickle.load(open("./datasets/prior_boxes_ssd300.pkl", 'rb'))
#         self.bbox_util = BBoxUtility(self.num_classes, prior)
#
#         self.pre_trained = "./ckpt/pre_trained/weights_SSD300.hdf5"
#
#         # 导入预训练模型进行微调
#         self.model = SSD300(self.input_shape, num_classes=self.num_classes)
#
#     def get_detection_data(self):
#         """
#         获取检测的迭代数据
#         :return:
#         """
#         # 1.获取标注数据，构造训练图片名字列表，测试图片名字列表
#         gt = pickle.load(open(self.gt_path, 'rb'))
#         # 图片名字列表
#         keys = sorted(gt.keys())
#         number = int(round(0.8 * len(keys)))
#         train_keys = keys[:number]
#         val_keys = keys[number:]
#
#         # 2. 通过generator去获取迭代批次数据
#         # gt: 所有数据的目标值字典
#         # path_prefix: 图片路径
#         # image_size: 转换成的固定图片大小
#         gen = Generator(gt, self.bbox_util, 16, self.image_path,
#                         train_keys, val_keys,
#                         (self.input_shape[0], self.input_shape[1]), do_crop=False)
#         return gen
#
#     def init_model_param(self):
#         """
#         初始化网络模型参数，指定微调的时候，训练部分
#         :return:
#         """
#         # 1、加载本地预训练好的模型
#         self.model.load_weights(self.pre_trained, by_name=True)
#         # 2、指定模型当中某些结构freeze
#         # 冻结模型部分 SSD当中的VGG前半部分
#         freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
#                   'conv2_1', 'conv2_2', 'pool2',
#                   'conv3_1', 'conv3_2', 'conv3_3', 'pool3']
#
#         for L in self.model.layers:
#             if L.name in freeze:
#                 L.trainable = False
#         return None
#
#     def compile(self):
#         """
#         编程模型，SSD网络的损失函数计算MultiboxLoss 的 compute_loss
#         :return:
#         """
#         # MultiboxLoss：N各类别+1背景类别
#         # TensorFlow.python.keras.optimizers.Adam() 出现问题，给4个，是需要3个参数
#         # keras 1.2.2 optimizers.Adam() 这个版本的函数可以
#         self.model.compile(optimizer=keras.optimizers.Adam(),
#                            loss=MultiboxLoss(self.num_classes).compute_loss)
#
#     def fit_generator(self, gen):
#         """
#         进行训练
#         :return:
#         """
#         # 建立一回调函数
#         callback = [ModelCheckpoint('./ckpt/fine_tuning/weights.{epoch:02d}-{val_acc:.2f}.hdf5',
#                                     monitor='val_acc',
#                                     save_best_only=True,
#                                     save_weights_only=True,
#                                     mode='auto',
#                                     period=1), TensorBoard(log_dir="./graph")]
#
#         self.model.fit_generator(gen.generate(True), gen.train_batches, self.epochs,
#                                  callbacks=callback, validation_data=gen.generate(False),
#                                  nb_val_samples=gen.val_batches)
#
# if __name__ == '__main__':
#     ssd = SSDTrain(num_classes=9)
#     gen = ssd.get_detection_data()
#     ssd.init_model_param()
#     ssd.compile()
#     ssd.fit_generator(gen)

import pickle
from utils.detection_generate import Generator
from utils.ssd_utils import BBoxUtility
from nets.ssd_net import SSD300
from utils.ssd_losses import MultiboxLoss
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
import keras
import tensorflow as tf


class SSDTrain(object):

    def __init__(self, num_classes=9, input_shape=(300, 300, 3), epochs=30):
        """初始化网络指定一些参数，训练数据类别，图片需要指定模型输入大小，迭代次数
        """
        self.num_classes = num_classes
        self.batch_size = 32
        self.input_shape = input_shape
        self.epochs = epochs

        # 指定训练和读取数据的相关参数
        self.gt_path = "./datasets/commodity_gt.pkl"
        self.image_path = "./datasets/commodity/JPEGImages/"

        prior = pickle.load(open("./datasets/prior_boxes_ssd300.pkl", 'rb'))
        self.bbox_util = BBoxUtility(self.num_classes, prior)

        self.pre_trained = "./ckpt/pre_trained/weights_SSD300.hdf5"

        # 初始化模型
        self.model = SSD300(self.input_shape, num_classes=self.num_classes)

    def get_detection_data(self):
        """
        获取检测的迭代数据
        :return:
        """
        # 1、读取标注数据，构造训练图片名字列表，测试图片名字列表
        # gt = pickle.load(open(self.gt_path, 'rb'))
        # print(gt)
        # # 图片名字列表
        # name_keys = sorted(gt.keys())
        # number = int(round(0.8 * len(name_keys)))
        # train_keys = name_keys[:number]
        # val_keys = name_keys[number:]
        #
        # # 2、通过generator去获取迭代批次数据
        # # gt：所有数据的目标值字典
        # # path_prefix:图片的路径
        # # image_size:转换成的固定图片大小
        # gen = Generator(gt, self.bbox_util, self.batch_size, self.image_path,
        #                 train_keys, val_keys, (self.input_shape[0], self.input_shape[1]), do_crop=False)
        # 获取标记数据，分成训练集与测试集
        gt = pickle.load(open(self.gt_path, 'rb'))
        keys = sorted(gt.keys())
        num_train = int(round(0.8 * len(keys)))
        train_keys = keys[:num_train]
        val_keys = keys[num_train:]

        # Generator获取数据
        gen = Generator(gt, self.bbox_util, 16, self.image_path,
                        train_keys, val_keys,
                        (self.input_shape[0], self.input_shape[1]), do_crop=False)

        return gen

    def init_model_param(self):
        """
        初始化网络模型参数，指定微调的时候，训练部分
        :return:
        """
        # 1、加载本地预训练好的模型
        self.model.load_weights(self.pre_trained, by_name=True)
        # 2、指定模型当中某些结构freeze
        # 冻结模型部分为 SSD当中的VGG前半部分
        freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
                  'conv2_1', 'conv2_2', 'pool2',
                  'conv3_1', 'conv3_2', 'conv3_3', 'pool3']

        for L in self.model.layers:
            if L.name in freeze:
                L.trainable = False

        return None

    def compile(self):
        """编译模型
        SSD网络的损失函数计算MultiboxLoss 的compute_loss
        """

        # MultiboxLoss:N个类别+1背景类别
        # TensorFlow.python.keras.optimizers.Adam() 出现问题，给4个，是需要3个参数
        # keras 1.2.2 optimizers.Adam()  这个版本的函数可以
        distribution = tf.contrib.distribute.MirroredStrategy()

        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss=MultiboxLoss(self.num_classes).compute_loss,
                           distribution=distribution)

    def fit_generator(self, gen):
        """
        进行训练
        :return:
        """
        # 建立一回调函数
        callback = [
            ModelCheckpoint('./ckpt/fine_tuning/weights.{epoch:02d}-{val_acc:.2f}.hdf5',
                                            monitor='val_acc',
                                            save_best_only=True,
                                            save_weights_only=True,
                                            mode='auto',
                                            period=1),
            TensorBoard(log_dir="./graph")
        ]

        # self.model.fit_generator(gen.generate(train=True), gen.train_batches, self.epochs,
        #                          callbacks=callback, validation_data=gen.generate(train=False),
        #                          nb_val_samples=gen.val_batches)
        self.model.fit_generator(gen.generate(True), gen.train_batches, self.epochs,
                                 callbacks=callback,
                                 validation_data=gen.generate(False),
                                 nb_val_samples=gen.val_batches)


if __name__ == '__main__':
    ssd = SSDTrain(num_classes=9)
    gen = ssd.get_detection_data()
    ssd.init_model_param()
    ssd.compile()
    ssd.fit_generator(gen)
