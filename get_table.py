import matplotlib.pyplot as plt
import numpy as np
import os

from scipy import stats
from get_result import get_data_by_gp

def get_p_value(arrA, arrB):
    a = np.array(arrA)
    b = np.array(arrB)
    t, p = stats.ttest_ind(a,b)
    return p

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

path_use = 0
path_dict = ["./logs"]
path = path_dict[path_use]

methods =  ["DNM_multiple_32_M20", "MLP_32", "MLP_64", "DNM_Linear_M3_M20"]
methods2 = ["MDNM", "MLP(32)", "MLP(64)", "DNM"]

patterns = ["Train Acc", "Train Loss", "Test Acc", "Test Loss"]

# 获取数据
data_names = os.listdir(path)
print(data_names)
data_keys = []
for data_name in data_names:
    dataname = data_name[:-5]
    if "EW" in dataname:
        dataname = dataname[:-2]
    data_keys.append(dataname)
print(data_keys)


f = open('table.txt', 'w')
dnm_data = 0
p_value = 0
for i in range(len(data_names)):
    print('\multicolumn{6}{c}{%s}'%data_keys[i]+"\\\\\hline", file=f)
    print("ALGORITHM \t& Train Acc (MEAN $\pm$ STD) \t& Train Loss (MEAN $\pm$ STD) \t& Test Acc (MEAN $\pm$ STD) \t& P-VALUE \t& Test Loss (MEAN $\pm$ STD) \\\\\hline", file=f)
    result_list = get_data_by_gp(path, data_names[i], methods)

    mean_q = []
    for i_m in range(len(methods)):
        d_mean = np.mean(result_list[i_m], axis=0)
        mean_q.append(d_mean)
    best_m = np.zeros(len(methods))
    for i_p in range(len(patterns)):
        maxOmin = mean_q[0][i_p]
        for i_m in range(1, len(methods)):
            if i_p == 0 or i_p == 2:
                if maxOmin<mean_q[i_m][i_p]: # find max
                    maxOmin = mean_q[i_m][i_p]
                    best_m[i_p] = i_m
            else:
                if maxOmin>mean_q[i_m][i_p]: # find max
                    maxOmin = mean_q[i_m][i_p]
                    best_m[i_p] = i_m
            

    # print
    for i_m in range(len(methods)):
        print(methods2[i_m]+'\t&', end='',file=f)
        for i_p in range(len(patterns)):
            d_mean = np.mean(result_list[i_m], axis=0)[i_p]
            d_std = np.std(result_list[i_m], axis=0)[i_p]
            if i_m == best_m[i_p]:
                if i_p == 2:
                    if i_m == 0:
                        p_value = '-'
                        print("\\textbf{{{:.2f} $\pm$ {:.2f}}} (\%)\t& {} \t&".format(d_mean*100, d_std*100, p_value), end='',file=f)
                    else:
                        p_value = get_p_value(result_list[i_p][0], result_list[i_p][i_m])
                        print("\\textbf{{{:.2f} $\pm$ {:.2f}}} (\%)\t& {:.2e} \t&".format(d_mean*100, d_std*100, p_value), end='',file=f)
                elif i_p == 0:
                    print("\\textbf{{{:.2f} $\pm$ {:.2f}}} (\%) \t&".format(d_mean*100, d_std*100), end='',file=f)
                elif i_p == len(patterns)-1:
                    print("\\textbf{{{:.4f} $\pm$ {:.2f}}} \t".format(d_mean, d_std), end='',file=f)
                else:
                    print("\\textbf{{{:.4f} $\pm$ {:.2f}}} \t&".format(d_mean, d_std), end='',file=f)
            else:   
                if i_p == 2:
                    if i_m == 0:
                        p_value = '-'
                        print("{:.2f} $\pm$ {:.2f} (\%)\t& {} \t&".format(d_mean*100, d_std*100, p_value), end='',file=f)
                    else:
                        p_value = get_p_value(result_list[i_p][0], result_list[i_p][i_m])
                        print("{:.2f} $\pm$ {:.2f} (\%)\t& {:.2e} \t&".format(d_mean*100, d_std*100, p_value), end='',file=f)
                elif i_p == 0:
                    print("{:.2f} $\pm$ {:.2f} (\%) \t&".format(d_mean*100, d_std*100), end='',file=f)
                elif i_p == len(patterns)-1:
                    print("{:.4f} $\pm$ {:.2f} \t".format(d_mean, d_std), end='',file=f)
                else:
                    print("{:.4f} $\pm$ {:.2f} \t&".format(d_mean, d_std), end='',file=f)
        if i_m == len(methods)-1:
            print('\\\\\hline',file=f)
        else:
            print('\\\\',file=f)


    '''d1 = np.mean(result_list[method_num][0])
    d2 = np.mean(result_list[method_num][1])
    d3 = np.mean(result_list[method_num][2])
    d4 = np.mean(result_list[method_num][3])
    d5 = np.mean(result_list[method_num][4])

    data_RDNN.append(d1)
    data_LSTM.append(d2)
    data_MLP.append(d3)
    data_DNM.append(d4)
    data_SVM.append(d5)
    
    print(data_name)
    print(d1)
    print(d2)
    print(d3)
    print(d4)
    print(d5)

    if np.isnan(d1) or np.isnan(d2):
        data_names.remove(data_name)
    else:
        data_RDNN.append(d1)
        data_LSTM.append(d2)
        data_MLP.append(d3)
        data_DNM.append(d4)
        print(data_name)
        print(d1)
        print(d2)
        print(d3)
        print(d4)'''

# with open("xdaf.txt",'w',encoding='utf-8') as f:
#     for i in range(len(data_DNM)):
#         f.write(str(data_names[i]))
#         f.write('\t')
#         f.write(str(data_RDNN[i]))
#         f.write('\t')
#         f.write(str(data_LSTM[i]))
#         f.write('\t')
#         f.write(str(data_MLP[i]))
#         f.write('\t')
#         f.write(str(data_DNM[i]))
#         f.write('\n')
#         f.write(str(data_SVM[i]))
#         f.write('\n')
