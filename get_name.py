import os

data_path = "./data4"
file_names = os.listdir(data_path)

f = open("run_DNM_RNN_MODEL.sh", "w", encoding="utf-8")
for file_name in file_names:
    file_path = os.path.join(data_path, file_name)
    # file_path = data_path + "/" + file_name
    f.write('python main.py -m "DNM_RNN_MODEL" -d '+'"'+file_path+'"'+' -n '+str(10)+'\n')

f.close()


# f = open("run_SVM.sh", "w", encoding="utf-8")
# for file_name in file_names:
#     file_path = os.path.join(data_path, file_name)
#     # file_path = data_path + "/" + file_name
#     f.write('python main_SVM.py -m "SVM" -d '+'"'+file_path+'"'+' -n '+str(1)+'\n')

# f.close()