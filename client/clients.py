#This implement the client class aims to intergate the trainging process.
from abc import abstractmethod
import logging
from client.base.baseTrainer import BaseTrainer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from copy import deepcopy
from typing import OrderedDict
import math

logger = logging.getLogger(__name__)

class Client:
    '''
    Client base class for a client in federated learning

    1.Just receive a cid for the uique toked when initialize
    2.download() method to get global model form the server
    3.Use train() to produce training 
    4.Test is no need in this senario 
    '''
    def __init__(
        self,
        client_id: str,
        dataloader: DataLoader,
        model: nn.Module,
        trainer: BaseTrainer,
        args: dict = {},
        test_dataloader: DataLoader = None,
        watermarks: dict = {},
        global_args: dict = {},
    ) -> None:
        self.model = deepcopy(model)
        self.client_id = client_id
        self.dataloader = dataloader
        self.trainer = trainer
        self.args = args
        self.global_args = global_args
        self.test_dataloader = test_dataloader
        self.watermarks = watermarks
        
    def set_model(self, model: nn.Module) -> None:
        self.model = deepcopy(model)
        
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        self.model.load_state_dict(state_dict=state_dict)
        
    def get_model_state_dict(self) -> OrderedDict:
        return self.model.state_dict()
    
    def show_train_result(self, epoch: int, ret_list: list):
        #get the ret_list from trainer and show the trainig result
        
        #Info Head
        logger.info(f"Epoch: {epoch}, client id {self.client_id}",)
        
        for ind, ret in enumerate(ret_list):
            result = f"Inner Epoch: {ind} "
            for key, value in ret.items():
                result += f"{key}: {value} "
            logger.info(result)
        
        return
        
    def test(self, epoch: int) -> dict:
        '''
        return dict
        : client_id
          epoch
          loss
          acc
        '''
        #test routine for image classification 
        if (self.test_dataloader == None):
            logger.warn("No test data")
            return 
        self.model.eval()
        total_loss = 0
        correct = 0
        num_data = 0
        predict_label = torch.tensor([]).to(self.args['device'])
        true_label = torch.tensor([]).to(self.args['device'])
        for batch_id, batch in enumerate(self.test_dataloader):
            data, targets = batch
            data, targets = data.to(self.args['device']), targets.to(self.args['device'])
            true_label = torch.cat((true_label, targets), 0)
            output = self.model(data)
            total_loss += torch.nn.functional.cross_entropy(output, targets,
                                                            reduction='sum').item()  # sum up batch loss
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            predict_label = torch.cat((predict_label, pred), 0)
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
            num_data += output.size(0)
        acc = 100.0 * (float(correct) / float(num_data))
        total_l = total_loss / float(num_data)
        ret = dict()
        ret['client_id'] = self.client_id
        ret['epoch'] = epoch
        ret['loss'] = total_l
        ret['acc'] = acc
        logger.info(f"client id {self.client_id} with inner epoch {ret['epoch']}, Loss: {total_l}, Acc: {acc}")
        return ret
    
    def sign_test(self, epoch: int):
        kwargs = self.watermarks
        ind = 0 
        avg_private = 0
        count_private = 0
        self.model.eval()
        with torch.no_grad():
            if kwargs != None:
                for m in kwargs:
                    if kwargs[m]['flag'] == True:
                        b = kwargs[m]['b']
                        M = kwargs[m]['M']

                        M = M.to(self.args['device'])
                        if ind == 0 or ind == 1:
                            signbit = self.model.features[int(m)].scale.view([1, -1]).mm(M).sign().to(self.args['device'])
                        if ind == 2 or ind == 3:
                            w = torch.mean(self.model.features[int(m)].conv.weight, dim=0)
                            signbit = w.view([1,-1]).mm(M).sign().to(self.args['device'])
                        #print(signbit)

                        privatebit = b
                        privatebit = privatebit.sign().to(self.args['device'])
                
                        # print(privatebit)
    
                        detection = (signbit == privatebit).float().mean().item()
                        avg_private += detection
                        count_private += 1

        if kwargs == None:
            avg_private = -1.0 # Watermark doesn't exist
        if count_private != 0:
            avg_private /= count_private
        ret = dict()
        ret['client_id'] = self.client_id
        ret['epoch']     = epoch
        ret['sign_acc']  = avg_private
        logger.info(f"client id {self.client_id} with inner epoch {epoch}, Sign Accuarcy: {avg_private}")
        return ret
    
    @abstractmethod
    def train(self, epoch: int):
        return


class BaseClient(Client):

    def train(self, epoch: int):
        # --- 这是新的、正确的学习率调度逻辑 ---

        # 1. 从参数中获取初始学习率和总通信轮次
        #    注意: 我们需要从 self.args（即train_args）和 self.watermarks（这里借用来传递global_args）获取参数
        #    为了简化，我们先假设总轮数是固定的，比如200轮
        initial_lr = self.args.get('lr', 0.01)
        total_rounds = self.global_args.get('communication_round', 200) # 假设在train_args中也定义了总轮数

        # 2. 手动计算当前全局轮次 (epoch) 对应的余弦退火学习率
        #    这是余弦退火的数学公式
        eta_min = 1e-6  # 学习率最小值
        current_lr = eta_min + 0.5 * (initial_lr - eta_min) * \
                    (1 + math.cos(math.pi * epoch / total_rounds))

        if epoch % 10 == 0: # 每10轮打印一次，方便观察
            logger.info(f"Client {self.client_id} at Global Round {epoch}, LR is {current_lr:.6f}")

        # 3. 将计算出的新学习率用于本轮训练
        train_args_for_this_round = self.args.copy()
        train_args_for_this_round['lr'] = current_lr

        cal = self.trainer(self.model, self.dataloader, torch.nn.CrossEntropyLoss(), train_args_for_this_round)
        ret_list = cal.train(self.args.get('local_epochs', 5)) # 本地仍然训练5轮

        self.show_train_result(epoch, ret_list)
        return
    

class SignClient(Client):
    # test the watermark accuarcy.
    
    def train(self, epoch: int, watermarks: dict = None):
        cal = self.trainer(self.model,self.dataloader,torch.nn.CrossEntropyLoss(), self.args, self.watermarks)
        ret_list = cal.train(self.args.get('num_steps'))
        self.show_train_result(epoch, ret_list)
        # avg_loss = ret['loss']
        # sign_loss = ret['sign_loss']
        # Loss: {avg_loss}, Sign Loss: {sign_loss}")
        return 
    