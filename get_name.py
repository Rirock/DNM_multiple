import os

data_path = "./Dataset"
file_names = os.listdir(data_path)
model_name = "MLP"  #"DNM_multiple"
run_times = 30
hidden_size = 64
DNM_M = 20

f = open("run_"+model_name+".sh", "w", encoding="utf-8")
for file_name in file_names:
    file_path = os.path.join(data_path, file_name)
    # file_path = data_path + "/" + file_name
    f.write('python train.py -m "'+model_name+'" -d '+'"'+file_path+'"'+' --hidden_size '+str(hidden_size)+' --DNM_M '+str(DNM_M)+' -n '+str(run_times)+'\n')

f.close()

