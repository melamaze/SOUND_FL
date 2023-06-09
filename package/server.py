from .config import for_FL as f
# from .FL.datasets import Dataset
from .FL.attackers import Attackers
from .FL.clients import Server
from .Voice.create_model import cnn_model
from .Voice.dataset import Dataset

from torchvision.models import resnet18


from datetime import datetime

import torch
import copy
import numpy as np
import time
import pdb

def main():

    #為了固定跑出的結果，指定seed
    np.random.seed(f.seed)
    torch.manual_seed(f.seed)

    print(torch.cuda.is_available())
    f.device = torch.device('cuda:{}'.format(f.gpu) if torch.cuda.is_available() and f.gpu != -1 else 'cpu')
    #看是否使用cuda gpu
    print(f.device)

    # 建一個Dataset物件
    my_data = Dataset()
    
    ## pdb.set_trace()
    FL_net = cnn_model().to(f.device)

    print('The model in server:')
    print(FL_net)

    # model的參數
    FL_weights = FL_net.state_dict()

    # 建一個 Attackers 物件
    my_attackers = Attackers()

    trigger=None
    if(f.attack_mode == "poison"):
        # 原本是設定有哪些是攻擊者，我把它移到choose attacker了
        # 現在這裡是生成trigger
        # size = [75, 85, 100]
        # pos = "start", "mid", "end"
        # cont = True, false
        trigger = my_attackers.poison_setting(85, "start", True)
    

    my_server  = Server(copy.deepcopy(FL_net))

    # 用於之後分給 clients
    all_users = [i for i in range(f.total_users)]
    # 打亂所有 user 的順序   
    idxs_users = np.random.choice(range(f.total_users), f.total_users, replace=False)   

    if(f.attack_mode == 'poison'):
        # 選定哪些為攻擊者
        my_attackers.choose_attackers(idxs_users, my_data)
        print("number of attacker: ", my_attackers.attacker_count)         
        print("all attacker: ", my_attackers.all_attacker)   
        print("")
        
    my_server.split_user_to(all_users, my_attackers.all_attacker)
    
    # 分割資料給每個user
    my_data.sampling_list(my_attackers.all_attacker, trigger)
  
    total_time = 0

    # 從現在到程式結束的時間
    true_start_time = time.time()

    # 此處之 epochs 即要跑多少輪
    # 一輪就是每個 user train 5 round，再把每個 user 的 model 合起來
    for round in range(f.epochs):

        # 每輪都要重置「模型參數」、「模型 loss」
        my_server.reset()

        # 各client跑local epoch的時間
        # 因為實際上跑的時候是sequence而非parallel
        # 要是能parallel更好
        global_test_time = 0

        start_ep_time = time.time()

        # user train 並 Aggregate 起來
        # user train 會需要用到各 user 分配到圖片，其中攻擊者的需要特別處理
        # 我們這邊是 poison attack(故意給錯的 label)，你們應該要改成加上 trigger 之類的(?
        # 應該主要是改 Update.py 裡 LocalUpdate_poison.train
        # if(f.attack_mode == "poison"):
        my_server.local_update_poison(my_data, my_attackers.all_attacker, round)
        
        end_ep_time = time.time()

        local_ep_time = end_ep_time - start_ep_time
        
        # 對client進行validation
        # 並取得所花的時間
        global_test_time += my_server.show_testing_result(my_data)
        

        print("local_ep_time: ",local_ep_time)
        # 跑 1 round 的時間 (validation的時間概念上是在central server進行，因此不以parallel來算)
        round_time = local_ep_time + global_test_time
        print("round_time: ",round_time)
        print("")
        print("-------------------------------------------------------------------------")
        print("")

        total_time += round_time

        # 把 global model 存起來
        path = f.model_path + 'global_model_44100_attModel' + '.pt1'
        torch.save(my_server.client_net.state_dict(), path)

    # 程式結束 (其實也就多了存model的時間)
    true_end_time = time.time()
    
    print('simulation total time:', total_time)
    print('true total time:', true_end_time - true_start_time)



