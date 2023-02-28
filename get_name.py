import os

data_path_list = ["./Datasets/real_data/MAT", "./Datasets2", "./Datasets/synthetic/MAT"]

model_names = ["DNM_Linear", "DNM_multiple", "MLP", "MLP", "ICFC"]#, "KNN"]

def get_train_name(model_name):
    # 生成训练脚本
    run_times = 10
    hidden_size = 32
    DNM_M = 20

    save_path = "run_"+model_name+".sh"
    if os.path.exists(save_path):
        os.remove(save_path)

    for data_path in data_path_list:
        file_names = os.listdir(data_path)

        f = open(save_path, "a", encoding="utf-8")
        for file_name in file_names:
            file_path = os.path.join(data_path, file_name)
            if "KNN" in model_name:
                f.write('python train_KNN.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' -n '+str(run_times)+'\n')
            else:
                f.write('python train_28.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' --hidden_size '+str(hidden_size)+' --DNM_M '+str(DNM_M)+' -n '+str(run_times)+'\n')
            #f.write('python train.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' --hidden_size '+str(hidden_size)+' --DNM_M '+str(DNM_M)+' -n '+str(run_times)+'\n')
            # f.write('python train_EA.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' --DNM_M '+str(DNM_M)+' -n '+str(run_times)+'\n')

        f.close()

def get_train_value_name(model_names):
    save_path = "run_all_v"+".sh"
    if os.path.exists(save_path):
        os.remove(save_path)
    run_times = 10
    DNM_M = 20
    
    count = 0
    for model_name in model_names:  #"DNM_multiple" "MLP" "ICFC" "KNN"
        hidden_size = 32
        if "MLP" == model_name:
            if count > 0:
                hidden_size = 64
            count += 1
        
        for data_path in data_path_list:
            file_names = os.listdir(data_path)

            f = open(save_path, "a", encoding="utf-8")
            for file_name in file_names:
                file_path = os.path.join(data_path, file_name)
                if "KNN" in model_name:
                    f.write('python train_KNN.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' -n '+str(run_times)+'\n')
                else:
                    f.write('python get_train_value.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' --hidden_size '+str(hidden_size)+' --DNM_M '+str(DNM_M)+' -n '+str(run_times)+'\n')
                #f.write('python train.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' --hidden_size '+str(hidden_size)+' --DNM_M '+str(DNM_M)+' -n '+str(run_times)+'\n')
                # f.write('python train_EA.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' --DNM_M '+str(DNM_M)+' -n '+str(run_times)+'\n')

            f.close()

def get_trainDNM_M_name(model_name):
    save_path = "run_all_DNM"+".sh"
    if os.path.exists(save_path):
        os.remove(save_path)
    run_times = 10
    hidden_size = 32
    DNM_M_LIST = range(2,21,2)
    for DNM_M in DNM_M_LIST:
        for data_path in data_path_list:
            file_names = os.listdir(data_path)
            f = open(save_path, "a", encoding="utf-8")
            for file_name in file_names:
                file_path = os.path.join(data_path, file_name)
                f.write('python train_28.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' --hidden_size '+str(hidden_size)+' --DNM_M '+str(DNM_M)+' -n '+str(run_times)+'\n')
            f.close()

#get_train_value_name(model_names)
# get_train_name(model_names[0])
get_trainDNM_M_name(model_names[0])