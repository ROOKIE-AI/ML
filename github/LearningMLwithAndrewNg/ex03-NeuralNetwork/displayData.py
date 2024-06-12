import matplotlib.pyplot as plt
import numpy as np


def display_data(x):
    (m, n) = x.shape
    # ÿ��������ʾ�Ŀ�Ⱥ͸߶�
    example_width = np.round(np.sqrt(n)).astype(int)
    example_height = (n / example_width).astype(int)
    # ������ʾ��ʽ100����������10��10����ʾ
    display_rows = np.floor(np.sqrt(m)).astype(int)
    display_cols = np.ceil(m / display_rows).astype(int)
    # ����ʾ��ÿ��ͼ֮��ļ��
    pad = 1
    # ��ʾ�Ĳ��־��� ��ʼ��ֵΪ-1
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                               pad + display_rows * (example_height + pad)))
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
            max_val = np.max(np.abs(x[curr_ex]))
            display_array[pad + j * (example_height + pad) + np.arange(example_height),
                          pad + i * (example_width + pad) + np.arange(example_width)[:, np.newaxis]] = \
                x[curr_ex].reshape((example_height, example_width)) / max_val
            curr_ex += 1
        if curr_ex > m:
            break
    # ��ʾͼƬ
    plt.figure()
    plt.imshow(display_array, cmap='gray', extent=[-1, 1, -1, 1])
    plt.axis('off')
    plt.show()
