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
from IC_model import *

device = torch.device('cuda:0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--train_model", default="ICFC", type=str, help="use model")
    parser.add_argument("-d", "--data_path", default="./Datasets/real_data/MAT\Glass_Identification.mat", type=str, help="data path")
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
    epochs = 20000
    learning_rate = 0.001
    BATCH_SIZE = 1024*4
    model_save_path = "./models"

    # load data
    data = sio.loadmat(data_path)


    train_data = data["Xtr"].T
    train_label = data["Ytr"].T-1
    test_data = data["Xtt"].T
    test_label = data["Ytt"].T-1

    log_path = os.path.join("./logs", data_name)
    if not os.path.exists(log_path):
        os.mkdir(log_path)


    train_num, input_size  = train_data.shape
    test_num, _ = test_data.shape
    
    out_size = np.max(train_label)+1


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
    elif "DNM_multiple" in model_name:
        net = eval(model_name+"(input_size, hidden_size, out_size, M).to(device)")
        log_path = os.path.join(log_path, model_name+"_"+str(hidden_size)+"_M"+str(M)+"_log.csv")
    elif "IC" in model_name:
        net = eval(model_name+"(input_size, hidden_size, out_size).to(device)")
        log_path = os.path.join(log_path, model_name+"_"+str(hidden_size)+"_log.csv")
    else:
        net = eval(model_name+"(input_size, out_size, M, device).to(device)")
        log_path = os.path.join(log_path, model_name+"_M"+str(M)+"_log.csv")

    for run in range(run_times):
        st = time.time()
        

        net.reset_parameters()
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            
            output = net(data)
            output = torch.squeeze(output) 
            loss = loss_func(output, target.squeeze().long())
            optimizer.zero_grad()  # 梯度信息清空
            loss.backward()  # 反向传播获取梯度
            optimizer.step()  # 优化器更新
            
            if epoch % 100 == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
            if epoch == (epochs - 1):
                correct = 0
                total = 0
                predicted=torch.max(output.data, 1)[1] # predicted=torch.argmax(output, 1)
                print(predicted.shape)
                print(target.shape)
                target = torch.squeeze(target)
                correct += (predicted == target).sum().item()
                total += target.size(0)
                train_acc.append(correct / total)
                train_loss.append(loss.item())


        # Testing
        test_fit = net(test_data)
        test_fit = torch.squeeze(test_fit) 
        predicted=torch.max(test_fit.data, 1)[1]
        correct = (predicted == test_target.squeeze()).sum().item()
        test_acc.append(correct / test_num)

        loss = loss_func(test_fit, test_target.squeeze().long())
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