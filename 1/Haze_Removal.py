import cv2
import numpy as np
import math
import os
from skimage.metrics import peak_signal_noise_ratio as psnr  
from skimage.metrics import structural_similarity as ssim  # 导入PSNR,SSIM用于图像质量评估
import time
import matplotlib.pyplot as plt

class HazeRemoval():
    def __init__(self, omega = 0.95, t0 = 0.1, radius = 7, r = 60, eps = 1e-4):
        self.omega = omega  # 透射率估计的权重参数
        self.t0 = t0  # 最小透射值
        self.radius = radius  # 暗通道计算半径
        self.r = r  # 导向滤波器半径
        self.eps = eps  # 导向滤波器的epsilon值

    def get_dark_channel(self, img):
        """计算暗通道"""
        b, g, r = cv2.split(img)
        min_rgb = cv2.min(cv2.min(r, g), b)  # 计算每个像素的RGB通道中的最小值
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * self.radius + 1, 2 * self.radius + 1))
        dark = cv2.erode(min_rgb, kernel)  # 进行最小滤波以获取暗通道
        return dark

    def estimate_atmospheric_light(self, img, dark):
        """估计大气光A"""
        [h,w] = img.shape[:2]
        imgsize = h * w
        Top_pixels = int(max(math.floor(h * w/1000), 1))  # 计算暗通道中最亮的前0.1%像素数量

        darkvec = dark.reshape(imgsize)
        imgvec = img.reshape(imgsize, 3)

        indexes = darkvec.argsort()
        indexes = indexes[imgsize - Top_pixels::]  # 选择对应于暗通道中最亮的前0.1%像素的索引

        A = np.zeros(3)
        for i in range(Top_pixels):
            pixel = imgvec[indexes[i], :]  # 获取所选索引处像素的RGB值
            for channel in range(img.shape[2]):
                if pixel[channel] > A[channel]:
                    A[channel] = pixel[channel]  # 如果当前像素值大于存储值,用当前像素值更新存储值
        return A

    def estimate_transmission(self, img, A):
        """估计透射率t"""
        I_dark = np.empty(img.shape, img.dtype)

        for channel in range(img.shape[2]):
            I_dark[:, :, channel] = img[:, :, channel] / A[channel]  # 将每个颜色通道除以相应的大气光值

        transmission = 1 - self.omega * self.get_dark_channel(I_dark)  #根据公式，使用暗通道先验估计透射率
        return transmission

    def guided_filter(self, img, p):
        """导向滤波"""
        m_I = cv2.boxFilter(img, cv2.CV_64F, (self.r, self.r))
        m_p = cv2.boxFilter(p, cv2.CV_64F, (self.r, self.r))
        m_Ip = cv2.boxFilter(img * p, cv2.CV_64F, (self.r, self.r))  # 均值滤波
        cov_Ip = m_Ip - m_I * m_p  # 协方差

        m_II = cv2.boxFilter(img * img, cv2.CV_64F, (self.r, self.r))  # 均值滤波
        var_I = m_II - m_I * m_I  # 方差

        a = cov_Ip / (var_I + self.eps)  # 系数 a
        b = m_p - a * m_I  # 系数 b

        m_a = cv2.boxFilter(a, cv2.CV_64F, (self.r, self.r))  # 对系数 a 进行均值滤波
        m_b = cv2.boxFilter(b, cv2.CV_64F, (self.r, self.r))  # 对系数 b 进行均值滤波

        q = m_a * img + m_b  # 计算导向滤波结果
        return q

    def refine_transmission(self, img, t):
        """透射率精炼"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray) / 255
        refined_t = self.guided_filter(gray, t)  # 对透射率进行导向滤波
        return refined_t

    def recover_image(self, img, t, A):
        """恢复去雾图像"""
        recovered_img = np.empty(img.shape, img.dtype)
        t = cv2.max(t, self.t0)  # 透射率取一个最小值

        for channel in range(img.shape[2]):
            recovered_img[:, :, channel] = (img[:, :, channel] - A[channel]) / t + A[channel]  # 恢复去雾图像

        return recovered_img * 255
    
    def recover_depth_map(self, transmission, A):
        """深度估计"""
        depth_map = -0.1 * np.log(transmission) * 255
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        depth_map_normalized_gray = (depth_map_normalized * 255).astype('uint8')
        depth_map_heatmap = cv2.applyColorMap(depth_map_normalized_gray, cv2.COLORMAP_HOT)
        return depth_map_heatmap

def plot_metrics(metric_list, metric_name, file_indexes, output_folder):
    plt.plot(file_indexes, metric_list, linestyle='-', label=metric_name)
    plt.xlabel('fig')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name}')
    plt.legend()
    plt.grid()
    
    output_file = os.path.join(output_folder, f"{metric_name}.png")
    plt.savefig(output_file)
    plt.close()


if __name__ == '__main__':

    origin_path = "./SOTS/HR_hazy"
    gt_path = "./SOTS/HR"
    hazedark_folder_path = './SOTS/HR_haze_dark'
    gtdark_folder_path = './SOTS/HR_gt_dark'
    recover_folder_path = "./SOTS/HR_recover"
    result_file_path = "./SOTS/result.txt"
    depth_folder_path = "./SOTS/Depth"

    files = os.listdir(origin_path)

    psnr_500 = []  # 储存 PSNR 值的列表
    psnr_1000 = []
    ssim_500 = []  # 储存 SSIM 值的列表
    ssim_1000 = []
    time_500 = []  # 储存耗时的列表
    time_1000 = []
    i = 0

    with open(result_file_path, "w") as f:

        for filename in files:

            file_path = os.path.join(origin_path, filename)
            gt_file_path = os.path.join(gt_path, filename)
            hazedark_file_path = os.path.join(hazedark_folder_path, filename.split(".")[0] + ".png")
            gtdark_file_path = os.path.join(gtdark_folder_path, filename.split(".")[0] + ".png")
            recover_file_path = os.path.join(recover_folder_path, filename.split(".")[0] + ".png")
            depth_file_path = os.path.join(depth_folder_path, filename.split(".")[0] + ".png")

            img = cv2.imread(file_path)
            gt_img = cv2.imread(gt_file_path)

            Haze = HazeRemoval()

            start_time = time.time()

            I = img.astype('float64') / 255  # 将雾图像归一化

            dark = Haze.get_dark_channel(I)  # 计算暗通道
            A = Haze.estimate_atmospheric_light(I, dark)  # 估计大气光
            t = Haze.estimate_transmission(I, A)  # 估计透射率
            refined_t = Haze.refine_transmission(img, t)  # 精炼透射率
            recovered_img = Haze.recover_image(I, refined_t, A)  # 恢复去雾图像
            
            end_time = time.time()
            elapsed_time = end_time - start_time

            depth_map = Haze.recover_depth_map(refined_t, A)

            i += 1
            
            cv2.imwrite(hazedark_file_path, dark * 255)
            cv2.imwrite(gtdark_file_path, Haze.get_dark_channel(gt_img.astype('float64') / 255) * 255)
            cv2.imwrite(recover_file_path, recovered_img)
            cv2.imwrite(depth_file_path, depth_map)

            psnr_val = psnr(gt_img, recovered_img, data_range=255)
            ssim_val = ssim(gt_img, recovered_img, channel_axis=2, data_range=255)

            if i <= 500:
                psnr_500.append(psnr_val)
                ssim_500.append(ssim_val)
                time_500.append(elapsed_time)
            else:
                psnr_1000.append(psnr_val)
                ssim_1000.append(ssim_val)
                time_1000.append(elapsed_time)

            print(f"File {filename}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, time={elapsed_time:.4f}s\n")
            f.write(f"File {filename}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, time={elapsed_time:.4f}s\n")

        psnr_500_avg = np.mean(psnr_500)
        ssim_500_avg = np.mean(ssim_500)
        time_500_avg = np.mean(time_500)

        psnr_1000_avg = np.mean(psnr_1000)
        ssim_1000_avg = np.mean(ssim_1000)
        time_1000_avg = np.mean(time_1000)

        print(f"Average PSNR Indoor : {psnr_500_avg:.2f}, Average PSNR Outdoor : {psnr_1000_avg:.2f}, Average SSIM Indoor : {ssim_500_avg:.4f}, Average SSIM Outdoor : {ssim_1000_avg:.4f}, Average time Indoor : {time_500_avg:.4f}s, Average time Outdoor : {time_1000_avg:.4f}s\n")
        f.write(f"Average PSNR Indoor : {psnr_500_avg:.2f}, Average PSNR Outdoor : {psnr_1000_avg:.2f}, Average SSIM Indoor : {ssim_500_avg:.4f}, Average SSIM Outdoor : {ssim_1000_avg:.4f}, Average time Indoor : {time_500_avg:.4f}s, Average time Outdoor : {time_1000_avg:.4f}s\n")

    # file_indexes = [i for i, _ in enumerate(files)]
    # output_folder = "./SOTS"
    # plot_metrics(psnr_list, 'PSNR', file_indexes, output_folder)
    # plot_metrics(ssim_list, 'SSIM', file_indexes, output_folder)
