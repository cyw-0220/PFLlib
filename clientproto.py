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

import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from collections import defaultdict

import torch.nn.functional as F


class clientProto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.tau = args.tau
        self.mu = args.mu

        self.protos = None  # 初始化当前客户端原型
        self.global_protos = None  # 全局训练原型
        self.old_protos = copy.deepcopy(self.protos)  # 存储上一次训练原型
        self.old_model = copy.deepcopy(self.model)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda


    def train(self):  # 训练函数

        trainloader = self.load_train_data()  # 加载训练数据
        start_time = time.time()  # 记录训练开始的时间

        self.model.train()   # 设置模型为训练模式


        max_local_epochs = self.local_epochs  # 设置最大本地训练轮数为类属性中定义的本地训练轮数

        if self.train_slow:  # 如果设置为慢速训练，就随机减少本地训练轮数
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # todo 此处默认字典？
        protos = defaultdict(list)  # 创建一个默认字典用来存储各个类别的原型向量

        for epoch in range(max_local_epochs):  # 以最大本地训练轮数为限制开始训练
            # if epoch > 0:
            #     self.old_protos = copy.deepcopy(self.protos)
            for i, (x, y) in enumerate(trainloader):  # 遍历训练加载器中的数据和标签
                if type(x) == type([]):  # 如果输入数据x是一个列表
                    x[0] = x[0].to(self.device)  # 只取第一个部分并放到指定设备上
                else:
                    x = x.to(self.device)  # 否则直接将输入数据放到指定设备上
                y = y.to(self.device)  # 将标签数据放到指定设备上
                if self.train_slow:  # 如果设置为慢速训练，程序会随机休眠一段时间
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep = self.model.base(x)  # 获取模型的基础部分的输出
                output = self.model.head(rep)  # 获取模型头部的输出
                loss = self.loss(output, y)  # 记录损失

                # 如果存在全局原型向量，将其与当前得到的原型向量整合
                if self.global_protos is not None and self.old_protos is not None:

                    old_rep = self.old_model.base(x).detach()
                    # global_rep = self.global_model.base(x).detach()

                    global_rep = copy.deepcopy(rep.detach())  # 先复制本轮学习到的向量，避免修改原始数据
                    for i, yy in enumerate(y):  # 遍历当前批次中的所有标签
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):  # 如果全局原型向量为这个类别的话，更新新的原型向量
                            global_rep[i, :] = self.global_protos[y_c].data

                    loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, global_rep) / self.tau) / (
                                torch.exp(F.cosine_similarity(rep, global_rep) / self.tau) + torch.exp(
                            F.cosine_similarity(rep, old_rep) / self.tau)))
                    loss += self.mu * torch.mean(loss_con)

                    # todo 更新loss的算法
                    # loss += self.loss_mse(solved_rep, rep) * self.lamda  # 将增加的全局原型向量的损失添加到总损失中

                # 遍历当前批次中的所有标签，将对应的原型向量存入protos字典中
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)


                self.optimizer.zero_grad()  # 清零之前的梯度
                loss.backward()  # 反向传播损失
                self.optimizer.step()  # 优化器更新模型参数

        self.protos = agg_func(protos)  # todo
        self.old_protos = copy.deepcopy(self.protos)  # 更新旧模型为当前模型，对于下一轮的对比学习使用

        if self.learning_rate_decay:   # 如果启用了学习率衰减策略，这里进行学习率调整
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1  # 更新和记录训练时间成本
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_protos(self, global_protos):
        self.global_protos = global_protos

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        
        if self.global_protos is not None:
            with torch.no_grad():
                for x, y in testloaderfull:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = self.model.base(x)

                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in self.global_protos.items():
                            if type(pro) != type([]):
                                output[i, j] = self.loss_mse(r, pro)

                    test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0

    def train_metrics(self):
        trainloader = self.load_train_data()  # 加载训练数据
        self.model.eval()  # 设置模型为评估模式，这将关闭dropout等

        train_num = 0  # 记录处理的样本数量
        losses = 0  # 记录累积的损失值

        # 评估模型 —— 在不跟踪梯度的情况下进行迭代，以避免计算不必要的操作
        with torch.no_grad():
            for x, y in trainloader:  # 数据加载器提供数据批次
                if type(x) == type([]):  # 和训练方法中一样，如果数据包含多个部分，只移动图像
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)  # 获取当前模型的表示和输出
                output = self.model.head(rep)
                loss = self.loss(output, y)  # 计算损失

                # 如果存在全局原型向量，将其与当前得到的原型向量整合
                if self.global_protos is not None and self.old_protos is not None:

                    old_rep = self.old_model.base(x).detach()
                    # global_rep = self.global_model.base(x).detach()

                    global_rep = copy.deepcopy(rep.detach())  # 先复制本轮学习到的向量，避免修改原始数据
                    for i, yy in enumerate(y):  # 遍历当前批次中的所有标签
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):  # 如果全局原型向量为这个类别的话，更新新的原型向量
                            global_rep[i, :] = self.global_protos[y_c].data

                    loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, global_rep) / self.tau) / (
                            torch.exp(F.cosine_similarity(rep, global_rep) / self.tau) + torch.exp(
                        F.cosine_similarity(rep, old_rep) / self.tau)))
                    loss += self.mu * torch.mean(loss_con)


                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos