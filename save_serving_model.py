import tensorflow as tf
from keras import backend as K
from nets.ssd_net import SSD300
import os


def export_serving_model(version=3, path="./serving_model/commodity/"):
    """导出模型到pb文件
    :return:
    """
    # 1、路径 + 模型名字 以及版本, commodity模型名称
    model_path = os.path.join(
        tf.compat.as_bytes(path),
        tf.compat.as_bytes(str(version))
    )

    # 2、调用模型，指定读取训练好的物体检测的模型h5文件
    model = SSD300((300, 300, 3), num_classes=9)
    model.load_weights("./ckpt/fine_tuning/weights.07-4.51.hdf5")
    print(model.input, model.inputs)
    print(model.output)

    # 3、tf.saved_model.simple_save导出
    # 会话，模型路径，输入输出
    with K.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            model_path,
            inputs={"images": model.input},  # 注意input和inputs区别
            outputs={model.output.name: model.output}
        )


if __name__ == '__main__':
    export_serving_model()