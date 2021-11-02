import numpy as np
import matplotlib.pyplot as plt


def WhiteBlance(img, mode):
    print('Entered func')
    """White balance processing (default is 1 mean, 2 perfect reflection, 3 grayscale world, 4 based image analysis "and color correction, 5 dynamic threshold)"""
    #  Read image
    b, g, r = cv2.split(img)
    #  Mean is three-channel
    h, w, c = img.shape
    if mode == 2:
        #  Perfect reflection white balance --- Relying on the Ratio value to choose and the largest area of ​​the brightness is not a white image effect.
        output_img = img.copy()
        sum_ = np.double() + b + g + r
        hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
        Y = 765
        num, key = 0, 0
        ratio = 0.01
        while Y >= 0:
            num += hists[Y]
            if num > h * w * ratio / 100:
                key = Y
                break
            Y = Y - 1

        sumkey = np.where(sum_ >= key)
        sum_b, sum_g, sum_r = np.sum(b[sumkey]), np.sum(g[sumkey]), np.sum(r[sumkey])
        times = len(sumkey[0])
        avg_b, avg_g, avg_r = sum_b / times, sum_g / times, sum_r / times

        maxvalue = float(np.max(output_img))
        output_img[:, :, 0] = output_img[:, :, 0] * maxvalue / int(avg_b)
        output_img[:, :, 1] = output_img[:, :, 1] * maxvalue / int(avg_g)
        output_img[:, :, 2] = output_img[:, :, 2] * maxvalue / int(avg_r)
    elif mode == 3:
        #  Grayscale world hypothesis
        b_avg, g_avg, r_avg = cv2.mean(b)[0], cv2.mean(g)[0], cv2.mean(r)[0]
        #  Need to adjust the gain of RGB components
        k = (b_avg + g_avg + r_avg) / 3
        kb, kg, kr = k / b_avg, k / g_avg, k / r_avg
        ba, ga, ra = b * kb, g * kg, r * kr

        output_img = cv2.merge([ba, ga, ra])
    elif mode == 4:
        #  Image analysis - based bias detection and color correction
        I_b_2, I_r_2 = np.double(b) ** 2, np.double(r) ** 2
        sum_I_b_2, sum_I_r_2 = np.sum(I_b_2), np.sum(I_r_2)
        sum_I_b, sum_I_g, sum_I_r = np.sum(b), np.sum(g), np.sum(r)
        max_I_b, max_I_g, max_I_r = np.max(b), np.max(g), np.max(r)
        max_I_b_2, max_I_r_2 = np.max(I_b_2), np.max(I_r_2)
        [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
        [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])
        b0 = np.uint8(u_b * (np.double(b) ** 2) + v_b * b)
        r0 = np.uint8(u_r * (np.double(r) ** 2) + v_r * r)
        output_img = cv2.merge([b0, g, r0])
    elif mode == 5:
        #  Dynamic threshold algorithm ---- white point detection and white point adjustment
        #  Only white point detection is not the same as white point as the perfect reflection algorithm, but is determined by another rule.
        def con_num(x):
            if x > 0:
                return 1
            if x < 0:
                return -1
            if x == 0:
                return 0

        yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        #  YUV space
        (y, u, v) = cv2.split(yuv_img)
        max_y = np.max(y.flatten())
        sum_u, sum_v = np.sum(u), np.sum(v)
        avl_u, avl_v = sum_u / (h * w), sum_v / (h * w)
        du, dv = np.sum(np.abs(u - avl_u)), np.sum(np.abs(v - avl_v))
        avl_du, avl_dv = du / (h * w), dv / (h * w)
        radio = 0.5  #  If the value is too small, the color temperature develops to the pole

        valuekey = np.where((np.abs(u - (avl_u + avl_du * con_num(avl_u))) < radio * avl_du)
                             | (np.abs(v - (avl_v + avl_dv * con_num(avl_v))) < radio * avl_dv))
        num_y, yhistogram = np.zeros((h, w)), np.zeros(256)
        num_y[valuekey] = np.uint8(y[valuekey])
        yhistogram = np.bincount(np.uint8(num_y[valuekey].flatten()), minlength=256)
        ysum = len(valuekey[0])
        Y = 255
        num, key = 0, 0
        while Y >= 0:
            num += yhistogram[Y]
            if num > 0.1 * ysum:  #  Take the first 10% highlights as the calculated value, if the value is too large, the value is too small to adjust the amplitude
                key = Y
                break
            Y = Y - 1

        sumkey = np.where(num_y > key)
        sum_b, sum_g, sum_r = np.sum(b[sumkey]), np.sum(g[sumkey]), np.sum(r[sumkey])
        num_rgb = len(sumkey[0])

        b0 = np.double(b) * int(max_y) / (sum_b / num_rgb)
        g0 = np.double(g) * int(max_y) / (sum_g / num_rgb)
        r0 = np.double(r) * int(max_y) / (sum_r / num_rgb)

        output_img = cv2.merge([b0, g0, r0])
    else:
        #  Default mean ---- Simple mean white balance method
        b_avg, g_avg, r_avg = cv2.mean(b)[0], cv2.mean(g)[0], cv2.mean(r)[0]
        #  Ask the gain of each channel
        k = (b_avg + g_avg + r_avg) / 3
        kb, kg, kr = k / b_avg, k / g_avg, k / r_avg
        b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        output_img = cv2.merge([b, g, r])
    output_img = np.uint8(np.clip(output_img, 0, 255))
    return output_img


#from skimage.io import imread, imshow
#from skimage import img_as_ubyte
from matplotlib.patches import Rectangle
import cv2
print('Hello')
frame = cv2.imread(r'C:\Users\Manoj\Documents\Fall21\473\color_correction\frame6.jpg')

mode =1
newimg = WhiteBlance(frame,mode)
cv2.imshow('White balanced',newimg)
cv2.imshow('Original',frame)
cv2.waitKey(0)