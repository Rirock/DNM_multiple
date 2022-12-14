import os
import argparse
import time
import torch
import scipy.io as sio
import numpy as np
import pandas as pd
from sko.DE import DE


from DNM_model.DNM_models import *

device = torch.device('cuda:0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--train_model", default="DNM_DE", type=str, help="use model")
    parser.add_argument("-d", "--data_path", default="./Dataset/SpectEW_data.mat", type=str, help="data path")
    parser.add_argument("--DNM_M", default=20, type=int, help="DNM M")
    parser.add_argument("-n", "--run_times", default=2, type=int, help="run times")
    parser.add_argument("-s", "--start_run_time", default=1, type=int, help="run times")

    args = parser.parse_args()

    model_name = getattr(args, 'train_model')
    data_path = getattr(args, 'data_path')
    M = getattr(args, 'DNM_M')
    run_times = getattr(args, 'run_times')
    start_run_time = getattr(args, 'start_run_time')

    path = os.getcwd()
    data_name = os.path.split(data_path)[-1].split(".")[0]

    # parameter
    model_save_path = "./models"

    # load data
    data = sio.loadmat(data_path)
    data1 = data['data']
    log_path = os.path.join("./logs", data_name)
    if not os.path.exists(log_path):
        os.mkdir(log_path)


    train_data = data1['train'][0][0]
    train_label = data1['trainLabel'][0][0][:,1]
    test_data = data1['test'][0][0]
    test_label = data1['testLabel'][0][0][:,1]

    train_num, input_size  = train_data.shape
    #I, J = train_data.shape
    test_num, _ = test_data.shape
    out_size = 1
    D = input_size * M * 2 # the dimension of sample

    # The parameter of DE
    popsize = 50 # the size of population
    F = 0.5 # scaling factor
    CR = 0.9 # crossover rate
    iter = 1000
    qs = 0.1
    k = 5 #2.5
    
    log_path = os.path.join(log_path, model_name+"_M"+str(M)+"_qs"+str(qs)+"_k"+str(k)+"_log.csv")

    def obj_func(p):
        w = np.array(p[:input_size*M])
        q = np.array(p[input_size*M:D])
        w = w.reshape(input_size,M)
        q = q.reshape(input_size,M)
        my_DNM = DNM(w=w, q=q, M=M, qs=qs, k=k)
        train_fit = my_DNM.run(train_data)
        train_fit = np.squeeze(train_fit)
        result = np.square(train_fit - train_label)
        
        return np.mean(result)

    def acc_DNM(data, label, p):
        w = np.array(p[:input_size*M])
        q = np.array(p[input_size*M:D])
        w = w.reshape(input_size,M)
        q = q.reshape(input_size,M)
        my_DNM = DNM(w=w, q=q, M=M, qs=qs, k=k)
        test_fit = my_DNM.run(data)
        test_fit = np.squeeze(test_fit)
        loss = np.mean((test_fit - label) ** 2) 
        
        test_fit[test_fit >= 0.5] = 1
        test_fit[test_fit < 0.5] = 0

        correct_prediction = np.equal(test_fit, label)
        
        return np.mean(correct_prediction), loss


    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    times = []
    model_wq = []

    for run in range(run_times):
        st = time.time()

        de = DE(func=obj_func, n_dim=D, size_pop=popsize, max_iter=iter, lb=[-1]*D, ub=[1]*D)
        best_x, best_y = de.run()

        #print('best_x:', best_x, '\n', 'best_y:', best_y)
        
        acc, loss = acc_DNM(train_data, train_label, best_x)
        train_acc.append(acc)
        train_loss.append(loss)

        acc, loss = acc_DNM(test_data, test_label, best_x)
        test_acc.append(acc)
        test_loss.append(loss)

        times.append(time.time() - st)
        model_wq.append(best_x)
        print(run+1, ' train_acc:', train_acc[run], ' train_loss:', train_loss[run], ' test_acc:', test_acc[run], ' test_loss:', test_loss[run], ' time:', times[run])

    save_path = os.path.join(model_save_path, data_name+"_"+model_name+".csv")
    save_model = pd.DataFrame(model_wq)
    save_model.to_csv(save_path, mode='w', header=False, index=False)

    log_list = [train_acc, train_loss, test_acc, test_loss, times]
    log_list = list(map(list, zip(*log_list))) # 转置
    save_log = pd.DataFrame(log_list)
    save_log.to_csv(log_path, mode='w', header=False, index=False)
    print('mean train:', np.mean(train_acc), ' test:', np.mean(test_acc))
    print()



