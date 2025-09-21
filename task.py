import logging
import hashlib
import json
from collections import OrderedDict

logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader
from config.algorithm import Algorithm
from server.aggregation_alg.fedavg import fedavgAggregator
from client.clients import Client, BaseClient, SignClient
from client.trainer.fedproxTrainer import fedproxTrainer
from client.trainer.SignTrainer import SignTrainer
from model.ModelFactory import ModelFactory
from dataset.DatasetFactory import DatasetFactory
from dataset.DatasetSpliter import DatasetSpliter
from chainfl.interact import chain_proxy


class Task:
    '''
    WorkFlow of a Task:
    0. Construct (Model, Dataset)--> Benchmark
    1. Construct (Server, Client)--> FL Algorithm
    3. Process of the dataset 
    '''
    def __init__(self, global_args: dict, train_args: dict, algorithm: Algorithm):
        self.global_args = global_args
        self.train_args = train_args
        self.model = None
        
        #Get Dataset
        # TODO pass the schema (object) instead of args directly.  
        logger.info("Constructing dataset %s from dataset Factory", global_args.get('dataset'))
        self.train_dataset = DatasetFactory().get_dataset(global_args.get('dataset'),True)
        self.test_dataset =  DatasetFactory().get_dataset(global_args.get('dataset'),False)
        #Get Model
        logger.info("Constructing Model from model factory with model %s and class_num %d", global_args['model'], global_args['class_num'])
        self.model = ModelFactory().get_model(model=self.global_args.get('model'),class_num=self.global_args.get('class_num'))
        
        #FL alg
        logger.info("Algorithm: {algorithm}")
        self.server = algorithm.get_server()
        self.server = self.server()
        self.trainer = algorithm.get_trainer()
        self.client = algorithm.get_client()
        
        #Get Client and Trainer
        self.client_list = None
        self.client_pool : list[Client] = []
        
    def __repr__(self) -> str:
        pass
    
    def _construct_dataloader(self):
        logger.info("Constructing dataloader with batch size %d, client_num: %d, non-iid: %s", self.global_args.get('batch_size')
                    , chain_proxy.get_client_num(), "True" if self.global_args['non-iid'] else "False")
        batch_size = self.global_args.get('batch_size')
        batch_size = 8 if (batch_size is None) else batch_size
        self.train_dataloader_list = DatasetSpliter().random_split(dataset     = self.train_dataset,
                                                                   client_list = chain_proxy.get_client_list(),
                                                                   batch_size  = batch_size)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=True)
    
    def _construct_sign(self):
        self.keys_dict = dict()
        self.keys = list()
        sign_num = self.global_args.get('sign_num')
        if(None == sign_num): 
            sign_num = 0
            logger.info("No client need to add watermark")
            for ind, (client_id,_) in enumerate(self.client_list.items()):
                self.keys_dict[client_id] = None
        else:
            logger.info(f"{sign_num} client(s) will inject watermark into their models")
            
            for i in range(self.global_args.get('client_num')):
                if i < self.global_args.get('sign_num'):
                    key = chain_proxy.construct_sign(self.global_args)
                    self.keys.append(key)
                else : 
                    self.keys.append(None)
            for ind, (client_id,_) in enumerate(self.client_list.items()):
                self.keys_dict[client_id] = self.keys[ind]
            #Project the watermake to the client TODO work with the blockchain
            #Get model Here better split another function.                 
            tmp_args = chain_proxy.construct_sign(self.global_args)
            self.model = ModelFactory().get_sign_model(model          = self.global_args.get('model'),
                                                       class_num      = self.global_args.get('class_num'),
                                                       in_channels    = self.global_args.get('in_channels'),
                                                       watermark_args = tmp_args)  
        return    
    
    def _regist_client(self):
        #Regist the client to the blockchain.
        for i in range(self.global_args['client_num']):
            chain_proxy.client_regist()
            print(f"----> 客户端 {i + 1} 注册完成，当前区块高度: {chain_proxy.get_block_height()}")
        self.client_list = chain_proxy.get_client_list()
    
    def _construct_client(self):
        for client_id, _ in self.client_list.items():
            new_client = self.client(client_id, self.train_dataloader_list[client_id], self.model, 
                                    self.trainer, self.train_args, self.test_dataloader, self.keys_dict[client_id])
            self.client_pool.append(new_client)
    
    def run(self):
        print("========> 步骤1: 正在注册客户端到区块链...")
        self._regist_client()
        print("========> 步骤2: 正在构建数据加载器 (Dataloader)...")
        self._construct_dataloader()
        print("========> 步骤3: 正在构建签名/水印...")
        self._construct_sign()
        print("========> 步骤4: 正在构建客户端实例...")
        self._construct_client()

        print("========> 步骤5: 进入主训练循环...")
        for i in range(self.global_args['communication_round']):
            for client in self.client_pool:
                client.train(epoch = i)
                client.test(epoch = i)
                client.sign_test(epoch = i)

            print(f"========> [第 {i + 1} 轮] 服务器接收上传...")
            self.server.receive_upload(self.client_pool)

            print(f"========> [第 {i + 1} 轮] 服务器聚合模型...")
            global_model = self.server.aggregate()

            # --- 从这里开始添加 ---
            # 1. 将模型参数字典转换为一个确定的字符串
            model_json = json.dumps(OrderedDict({k: v.tolist() for k, v in global_model.items()}))
            # 2. 计算SHA-256哈希值
            model_hash_hex = hashlib.sha256(model_json.encode('utf-8')).hexdigest()
            # 3. 将十六进制的哈希值转换为一个大整数
            model_hash_int = int(model_hash_hex, 16)
            # 4. 调用我们的新函数上传到区块链
            chain_proxy.upload_model_hash(model_hash_int, i + 1)
            print(f"----> 当前区块高度: {chain_proxy.get_block_height()}")
            # --- 添加结束 ---


            print(f"========> [第 {i + 1} 轮] 向客戶端分发新模型...")
            print(f"------------------------------------------------------------------")
            for client in self.client_pool:
                client.load_state_dict(global_model)