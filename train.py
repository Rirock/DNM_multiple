import os
import argparse
import time
import torch
import scipy.io as sio
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch.utils.data as Data 
from sklearn.model_selection import train_test_split

from DNM_model.DNM_models import *

device = torch.device('cuda:0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--train_model", default="DNM_multiple", type=str, help="use model")
    parser.add_argument("-d", "--data_path", default="./Dataset/SpectEW_data.mat", type=str, help="data path")
    parser.add_argument("--hidden_size", default=32, type=int, help="hidden size")
    parser.add_argument("--DNM_M", default=20, type=int, help="DNM M")
    parser.add_argument("-n", "--run_times", default=10, type=int, help="run times")
    parser.add_argument("-s", "--start_run_time", default=1, type=int, help="run times")

    args = parser.parse_args()

    model_name = getattr(args, 'train_model')
    data_path = getattr(args, 'data_path')
    hidden_size = getattr(args, 'hidden_size')
    M = getattr(args, 'DNM_M')
    run_times = getattr(args, 'run_times')
    start_run_time = getattr(args, 'start_run_time')

    path = os.getcwd()
    data_name = os.path.split(data_path)[-1].split(".")[0]

    # parameter
    epochs = 5000
    learning_rate = 0.001
    BATCH_SIZE = 1024
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
    test_num, _ = test_data.shape
    out_size = 1


    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    times = []

    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)
    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label)
    data, target = train_data.to(device).float(), train_label.to(device).float()
    test_data, test_target = test_data.to(device).float(), test_label.to(device).float()

    if model_name == "MLP":
        net = eval(model_name+"(input_size, hidden_size, out_size).to(device)")
        log_path = os.path.join(log_path, model_name+"_"+str(hidden_size)+"_log.csv")
    elif model_name == "DNM_multiple":
        net = eval(model_name+"(input_size, hidden_size, out_size, M).to(device)")
        log_path = os.path.join(log_path, model_name+"_"+str(hidden_size)+"_M"+str(M)+"_log.csv")
    else:
        net = eval(model_name+"(input_size, out_size, M, device).to(device)")
        log_path = os.path.join(log_path, model_name+"_M"+str(M)+"_log.csv")

    for run in range(run_times):
        st = time.time()

        net.reset_parameters()
        
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            logits = net(data)
            logits = torch.squeeze(logits) 
            loss = torch.mean((logits - target) ** 2) # MSE
            optimizer.zero_grad()  # 梯度信息清空
            loss.backward()  # 反向传播获取梯度
            optimizer.step()  # 优化器更新
            if epoch % 100 == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
            if epoch == (epochs - 1):
                tensor_one = torch.ones(train_num).cuda()
                tensor_zero = torch.zeros(train_num).cuda()
                classifi = torch.where(logits > 0.5, tensor_one, tensor_zero)
                train_acc.append(torch.sum(classifi.eq(target)).item() / len(classifi))
                train_loss.append(loss.item())
                
        # Testing
        test_fit = net(test_data)
        test_fit = torch.squeeze(test_fit) 
        test_fit = torch.where(test_fit > 0.5, torch.ones(test_num).cuda(), torch.zeros(test_num).cuda())
        test_acc.append(torch.sum(test_fit.eq(test_target)).item() / len(test_fit))
        loss = torch.mean((test_fit - test_target) ** 2)
        test_loss.append(loss.item())

        save_path = os.path.join(model_save_path, data_name+"_"+model_name+"_"+str(run)+".pth")
        torch.save(net.state_dict(), save_path)
        times.append(time.time() - st)
        print(run+1, ' train_acc:', train_acc[run], ' train_loss:', train_loss[run], ' test_acc:', test_acc[run], ' test_loss:', test_loss[run], ' time:', times[run])


    log_list = [train_acc, train_loss, test_acc, test_loss, times]
    log_list = list(map(list, zip(*log_list))) # 转置
    save_log = pd.DataFrame(log_list)
    save_log.to_csv(log_path, mode='w', header=False, index=False)
    print('mean train:', np.mean(train_acc), ' test:', np.mean(test_acc))
    print()