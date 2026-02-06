import os
import random
import shutil

if __name__ == '__main__':
    # 对应其中类别
    classes = ['NE','HA','AN','DI','FE','SA','SU']
    folder_names = ['train', 'test']
    # 未划分的数据集地址
    src_data_folder = "../../../Datasets/JAFFE/jaffe"
    # 划分后的数据集保存地址
    target_data_folder = "../../../Datasets/JAFFE/jaffe_split"
    # 划分比例
    train_scale = 0.7
    # 在目标目录下创建训练集和验证集文件夹
    for folder_name in folder_names:
        folder_path = os.path.join(target_data_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        # 在folder_path目录下创建类别文件夹
        for class_name in classes:
            class_folder_path = os.path.join(folder_path, class_name)
            os.mkdir(class_folder_path)

    # 获取所有的图片
    files = os.listdir(src_data_folder)  # 得到的是文件名列表
    # print(files)
    data = [file for file in files if file.endswith('tiff')]
    # 随机打乱图片顺序
    random.shuffle(data)
    # 统计保存各类图片数量
    class_sum = dict.fromkeys(classes, 0)
    for file in data:
        class_sum[file[3:5]] += 1  # 文件名格式：KA.AN1.39.tiff，其中AN表示类别

    # 记录训练集各类别图片的个数
    class_train = dict.fromkeys(classes, 0)
    # 记录测试集各类别图片的个数
    class_test = dict.fromkeys(classes, 0)
    # 遍历每个图片划分train/test
    for file in data:
        # 得到原图片目录地址
        src_img_path = os.path.join(src_data_folder, file)
        # 如果训练集中该类别个数未达划分数量，则复制图片并分配进入训练集
        if class_train[file[3:5]] < class_sum[file[3:5]]*train_scale:
            target_img_path = os.path.join(os.path.join(target_data_folder, 'train'), file[3:5])
            shutil.copy2(src_img_path, target_img_path)  # 复制图片从src_img_path到target_img_path
            class_train[file[3:5]] += 1
        # 否则，进入测试集
        else:
            target_img_path = os.path.join(os.path.join(target_data_folder, 'test'), file[3:5])
            shutil.copy2(src_img_path, target_img_path)
            class_test[file[3:5]] += 1

    # 输出标明数据集划分情况
    for class_name in classes:
        print("-" * 10)
        print("{}类共{}张图片,划分完成:".format(class_name, class_sum[class_name]))
        print("训练集：{}张，测试集：{}张".format(class_train[class_name], class_test[class_name]))
        # 基本是训练集：21~23张，测试集：9张这样的比例划分