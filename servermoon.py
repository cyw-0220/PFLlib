# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import time

from system.flcore.clients.clientmoon import clientMOON
from system.flcore.servers.serverbase import Server
from system.utils.data_utils import read_client_data
from threading import Thread


class MOON(Server):  # MOON继承自server类
    def __init__(self, args, times):  # 初始化
        super().__init__(args, times)  # 调用基类的初始化方法

        # select slow clients
        self.set_slow_clients()  # 设置慢速客户端
        self.set_clients(clientMOON)  # 设置客户端类型为 clientMOON

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []  # 初始化存储每一轮耗时的列表


    def train(self):  # 客户端训练
        for i in range(self.global_rounds+1):  # 进行全局轮数次训练
            s_t = time.time()  # 记录开始时间
            self.selected_clients = self.select_clients()  # 选择参与本轮训练的客户端
            self.send_models()  # 向所选客户端发送全局模型参数

            if i%self.eval_gap == 0:  # 如果当前轮次是评估间隔的倍数，则执行评估
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()  # 评估全局模型

            for client in self.selected_clients:
                client.train()  # 所选客户端训练

            # 使用多线程的客户端训练代码，可以选择使用，可能会加快计算
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()  # 接收来自客户端的更新后的模型参数
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)   # 如果启用DLG评估，并且满足评估频率，调用DLG评估方法
            self.aggregate_parameters()  # 聚合来自不同客户端的参数

            self.Budget.append(time.time() - s_t)  # 记录当前轮的训练时间，并将其添加到Budget列表中
            print('-'*50, self.Budget[-1])

            # 如果设置了自动中断，并且满足check_done方法的条件（可能是指模型性能达到某个阈值），则跳出训练循环
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nBest local accuracy.")
        print("\nAveraged time per iteration.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()  # 保存训练结果
        self.save_global_model()  # 保存全局训练模型

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientMOON)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
