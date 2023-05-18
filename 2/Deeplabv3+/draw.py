import matplotlib.pyplot as plt

def draw_loss(modelpath):
    # 读取train.txt文件中的数据
    filepath = './trainfile/' + modelpath + "/train.txt"
    with open(filepath, 'r') as f:
        data = f.readlines()

    # 分割数据
    itrs = []
    loss = []
    for item in data:
        items = item.strip().split(',')
        itrs.append(int(float(items[1].strip().split('/')[0].split(' ')[1])))
        loss.append(float(items[2].split('=')[1]))

    # 创建图形
    plt.plot(itrs, loss, color='red', linewidth=2)

    # 设置图形标题和标签
    plt.title("Loss Curve", fontsize=24)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Loss", fontsize=14)

    # 设置刻度标记的大小
    plt.tick_params(axis='both', labelsize=14)

    # 保存图形到文件
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')

def draw_val(modelpath):
    # 打开文件并读取数据
    filepath = './trainfile/' + modelpath + "/val.txt"

    itrs = list(range(100, 30001, 100))

    with open(filepath, 'r') as f:
        # 分割数据
        Overall_Acc = []
        Mean_Acc = []
        FreqW_Acc = []
        Mean_IoU = []
        Class_IoU_0 = []
        Class_IoU_1 = []
        Class_IoU_2 = []
        Class_IoU_3 = []
        Class_IoU_4 = []
        Class_IoU_5 = []
        Class_IoU_6 = []
        Class_IoU_7 = []
        Class_IoU_8 = []
        Class_IoU_9 = []
        Class_IoU_10 = []
        Class_IoU_11 = []
        Class_IoU_12 = []
        Class_IoU_13 = []
        Class_IoU_14 = []
        Class_IoU_15 = []
        Class_IoU_16 = []
        Class_IoU_17 = []
        Class_IoU_18 = []

        for line in f:
            # 将字符串转换为字典
            data = eval(line.strip())

            # 提取 Class IoU 中的键值对
            class_iou = data.get('Class IoU', {})
            class_iou_items = class_iou.items()

            # 创建新字典，包含所有原始字典中的键值对和 Class IoU 的键值对
            data.pop('Class IoU', None)
            new_data = dict(data)
            new_data.update(class_iou_items)

            # 将每个值分别添加到对应的列表中
            Overall_Acc.append(new_data.get('Overall Acc', 0))
            Mean_Acc.append(new_data.get('Mean Acc', 0))
            FreqW_Acc.append(new_data.get('FreqW Acc', 0))
            Mean_IoU.append(new_data.get('Mean IoU', 0))
            Class_IoU_0.append(new_data.get(0, 0))
            Class_IoU_1.append(new_data.get(1, 0))
            Class_IoU_2.append(new_data.get(2, 0))
            Class_IoU_3.append(new_data.get(3, 0))
            Class_IoU_4.append(new_data.get(4, 0))
            Class_IoU_5.append(new_data.get(5, 0))
            Class_IoU_6.append(new_data.get(6, 0))
            Class_IoU_7.append(new_data.get(7, 0))
            Class_IoU_8.append(new_data.get(8, 0))
            Class_IoU_9.append(new_data.get(9, 0))
            Class_IoU_10.append(new_data.get(10, 0))
            Class_IoU_11.append(new_data.get(11, 0))
            Class_IoU_12.append(new_data.get(12, 0))
            Class_IoU_13.append(new_data.get(13, 0))
            Class_IoU_14.append(new_data.get(14, 0))
            Class_IoU_15.append(new_data.get(15, 0))
            Class_IoU_16.append(new_data.get(16, 0))
            Class_IoU_17.append(new_data.get(17, 0))
            Class_IoU_18.append(new_data.get(18, 0))

    # 将每个列表放入一个列表中，以便作为参数传入
    data_dict = {
        'Overall_Acc': Overall_Acc,
        'Mean_Acc': Mean_Acc,
        'FreqW_Acc': FreqW_Acc,
        'Mean_IoU': Mean_IoU
        # 'Class_IoU_0': Class_IoU_0,
        # 'Class_IoU_1': Class_IoU_1,
        # 'Class_IoU_2': Class_IoU_2,
        # 'Class_IoU_3': Class_IoU_3,
        # 'Class_IoU_4': Class_IoU_4,
        # 'Class_IoU_5': Class_IoU_5,
        # 'Class_IoU_6': Class_IoU_6,
        # 'Class_IoU_7': Class_IoU_7,
        # 'Class_IoU_8': Class_IoU_8,
        # 'Class_IoU_9': Class_IoU_9,
        # 'Class_IoU_10': Class_IoU_10,
        # 'Class_IoU_11': Class_IoU_11,
        # 'Class_IoU_12': Class_IoU_12,
        # 'Class_IoU_13': Class_IoU_13,
        # 'Class_IoU_14': Class_IoU_14,
        # 'Class_IoU_15': Class_IoU_15,
        # 'Class_IoU_16': Class_IoU_16,
        # 'Class_IoU_17': Class_IoU_17,
        # 'Class_IoU_18': Class_IoU_18
    }

    for key, value in data_dict.items():
        # 绘制图像
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.plot(itrs, value)
        plt.xlabel("Iterations")
        plt.ylabel(key)
        plt.title(f'{key} Curve')

        # 保存图像
        plt.savefig(f'{key}.png', dpi=600)
        
        # 清除当前画布，为下一张图做准备
        plt.clf()

    # 定义颜色
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:brown', 'tab:pink',
            'tab:olive', 'tab:purple', 'lime', 'cyan', 'silver', 'gold', 'crimson', 'navy', 'maroon']

    class_iou_dict = {'Class_IoU_0': Class_IoU_0,
                    'Class_IoU_1': Class_IoU_1,
                    'Class_IoU_2': Class_IoU_2,
                    'Class_IoU_3': Class_IoU_3,
                    'Class_IoU_4': Class_IoU_4,
                    'Class_IoU_5': Class_IoU_5,
                    'Class_IoU_6': Class_IoU_6,
                    'Class_IoU_7': Class_IoU_7,
                    'Class_IoU_8': Class_IoU_8,
                    'Class_IoU_9': Class_IoU_9,
                    'Class_IoU_10': Class_IoU_10,
                    'Class_IoU_11': Class_IoU_11,
                    'Class_IoU_12': Class_IoU_12,
                    'Class_IoU_13': Class_IoU_13,
                    'Class_IoU_14': Class_IoU_14,
                    'Class_IoU_15': Class_IoU_15,
                    'Class_IoU_16': Class_IoU_16,
                    'Class_IoU_17': Class_IoU_17,
                    'Class_IoU_18': Class_IoU_18}

    # 画图
    fig, ax = plt.subplots(figsize=(20, 15))
    
    for key, value in class_iou_dict.items():
        plt.plot(itrs, value, color=colors[int(key.split('_')[-1])], label=key)

    # 添加标题和标签
    plt.title('Class IoU')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()

    plt.savefig(f'Class_IoU.png', dpi=600)

modelpath = 'resnet'

draw_val(modelpath)
