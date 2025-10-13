# import logging
# import hashlib
# import json
# from collections import OrderedDict
#
# logger = logging.getLogger(__name__)
#
# from torch.utils.data import DataLoader
# from config.algorithm import Algorithm
# from server.aggregation_alg.fedavg import fedavgAggregator
# from client.clients import Client, BaseClient, SignClient
# from client.trainer.fedproxTrainer import fedproxTrainer
# from client.trainer.SignTrainer import SignTrainer
# from model.ModelFactory import ModelFactory
# from dataset.DatasetFactory import DatasetFactory
# from dataset.DatasetSpliter import DatasetSpliter
# from chainfl.interact import chain_proxy
#
#
# class Task:
#     '''
#     WorkFlow of a Task:
#     0. Construct (Model, Dataset)--> Benchmark
#     1. Construct (Server, Client)--> FL Algorithm
#     3. Process of the dataset
#     '''
#     def __init__(self, global_args: dict, train_args: dict, algorithm: Algorithm):
#         self.global_args = global_args
#         self.train_args = train_args
#         self.model = None
#
#         #Get Dataset
#         # TODO pass the schema (object) instead of args directly.
#         logger.info("Constructing dataset %s from dataset Factory", global_args.get('dataset'))
#         self.train_dataset = DatasetFactory().get_dataset(global_args.get('dataset'),True)
#         self.test_dataset =  DatasetFactory().get_dataset(global_args.get('dataset'),False)
#         #Get Model
#         logger.info("Constructing Model from model factory with model %s and class_num %d", global_args['model'], global_args['class_num'])
#         self.model = ModelFactory().get_model(model=self.global_args.get('model'),class_num=self.global_args.get('class_num'))
#
#         #FL alg
#         logger.info("Algorithm: {algorithm}")
#         self.server = algorithm.get_server()
#         self.server = self.server()
#         self.trainer = algorithm.get_trainer()
#         self.client = algorithm.get_client()
#
#         #Get Client and Trainer
#         self.client_list = None
#         self.client_pool : list[Client] = []
#
#     def __repr__(self) -> str:
#         pass
#
#     def _construct_dataloader(self):
#         logger.info("Constructing dataloader with batch size %d, client_num: %d, non-iid: %s", self.global_args.get('batch_size')
#                     , chain_proxy.get_client_num(), "True" if self.global_args['non-iid'] else "False")
#         batch_size = self.global_args.get('batch_size')
#         batch_size = 8 if (batch_size is None) else batch_size
#         self.train_dataloader_list = DatasetSpliter().random_split(dataset     = self.train_dataset,
#                                                                    client_list = chain_proxy.get_client_list(),
#                                                                    batch_size  = batch_size)
#         self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=True)
#
#     def _construct_sign(self):
#         self.keys_dict = dict()
#         self.keys = list()
#         sign_num = self.global_args.get('sign_num')
#         if(None == sign_num):
#             sign_num = 0
#             logger.info("No client need to add watermark")
#             for ind, (client_id,_) in enumerate(self.client_list.items()):
#                 self.keys_dict[client_id] = None
#         else:
#             logger.info(f"{sign_num} client(s) will inject watermark into their models")
#
#             for i in range(self.global_args.get('client_num')):
#                 if i < self.global_args.get('sign_num'):
#                     key = chain_proxy.construct_sign(self.global_args)
#                     self.keys.append(key)
#                 else :
#                     self.keys.append(None)
#             for ind, (client_id,_) in enumerate(self.client_list.items()):
#                 self.keys_dict[client_id] = self.keys[ind]
#             #Project the watermake to the client TODO work with the blockchain
#             #Get model Here better split another function.
#             tmp_args = chain_proxy.construct_sign(self.global_args)
#             self.model = ModelFactory().get_sign_model(model          = self.global_args.get('model'),
#                                                        class_num      = self.global_args.get('class_num'),
#                                                        in_channels    = self.global_args.get('in_channels'),
#                                                        watermark_args = tmp_args)
#         return
#
#     def _regist_client(self):
#         #Regist the client to the blockchain.
#         for i in range(self.global_args['client_num']):
#             chain_proxy.client_regist()
#             print(f"----> 客户端 {i + 1} 注册完成，当前区块高度: {chain_proxy.get_block_height()}")
#         self.client_list = chain_proxy.get_client_list()
#
#     def _construct_client(self):
#         for client_id, _ in self.client_list.items():
#             new_client = self.client(client_id, self.train_dataloader_list[client_id], self.model,
#                                     self.trainer, self.train_args, self.test_dataloader, self.keys_dict[client_id])
#             self.client_pool.append(new_client)
#
#     def run(self):
#         print("========> 步骤1: 正在注册客户端到区块链...")
#         self._regist_client()
#         print("========> 步骤2: 正在构建数据加载器 (Dataloader)...")
#         self._construct_dataloader()
#         print("========> 步骤3: 正在构建签名/水印...")
#         self._construct_sign()
#         print("========> 步骤4: 正在构建客户端实例...")
#         self._construct_client()
#
#         print("========> 步骤5: 进入主训练循环...")
#         for i in range(self.global_args['communication_round']):
#             for client in self.client_pool:
#                 client.train(epoch = i)
#                 client.test(epoch = i)
#                 client.sign_test(epoch = i)
#
#             print(f"========> [第 {i + 1} 轮] 服务器接收上传...")
#             self.server.receive_upload(self.client_pool)
#
#             print(f"========> [第 {i + 1} 轮] 服务器聚合模型...")
#             global_model = self.server.aggregate()
#
#             # --- 从这里开始添加 ---
#             # 1. 将模型参数字典转换为一个确定的字符串
#             model_json = json.dumps(OrderedDict({k: v.tolist() for k, v in global_model.items()}))
#             # 2. 计算SHA-256哈希值
#             model_hash_hex = hashlib.sha256(model_json.encode('utf-8')).hexdigest()
#             # 3. 将十六进制的哈希值转换为一个大整数
#             model_hash_int = int(model_hash_hex, 16)
#             # 4. 调用我们的新函数上传到区块链
#             chain_proxy.upload_model_hash(model_hash_int, i + 1)
#             print(f"----> 当前区块高度: {chain_proxy.get_block_height()}")
#             # --- 添加结束 ---
#
#
#             print(f"========> [第 {i + 1} 轮] 向客戶端分发新模型...")
#             print(f"------------------------------------------------------------------")
#             for client in self.client_pool:
#                 client.load_state_dict(global_model)
# task.py

import logging
import hashlib
import json
import os
import binascii
from collections import OrderedDict

logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader
from config.algorithm import Algorithm
# from server.aggregation_alg.fedavg import fedavgAggregator # <-- 不再需要
from server.serverSimulator import serverSimulator  # <-- 导入新版服务器
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

        # Get Dataset
        logger.info("Constructing dataset %s from dataset Factory", global_args.get('dataset'))
        self.train_dataset = DatasetFactory().get_dataset(global_args.get('dataset'), True)
        self.test_dataset = DatasetFactory().get_dataset(global_args.get('dataset'), False)

        # Get Model
        logger.info("Constructing Model from model factory with model %s and class_num %d", global_args['model'],
                    global_args['class_num'])
        self.model = ModelFactory().get_model(model=self.global_args.get('model'),
                                              class_num=self.global_args.get('class_num'))

        # --- 核心修改：初始化新的TEE服务器 ---
        logger.info("Algorithm: {algorithm}")
        # self.server = algorithm.get_server() # <-- 废弃旧的服务器逻辑
        # self.server = self.server()         # <-- 废弃旧的服务器逻辑
        self.server = serverSimulator(client_num=self.global_args['client_num'])
        self.server._set_global_model(self.model)  # 将初始模型设置给服务器
        # --- 修改结束 ---

        self.trainer = algorithm.get_trainer()
        self.client = algorithm.get_client()

        self.client_list = None
        self.client_pool: list[Client] = []

    def __repr__(self) -> str:
        pass

    def _construct_dataloader(self):
        logger.info("Constructing dataloader with batch size %d, client_num: %d, non-iid: %s",
                    self.global_args.get('batch_size')
                    , chain_proxy.get_client_num(), "True" if self.global_args['non-iid'] else "False")
        batch_size = self.global_args.get('batch_size')
        batch_size = 8 if (batch_size is None) else batch_size
        self.train_dataloader_list = DatasetSpliter().random_split(dataset=self.train_dataset,
                                                                   client_list=chain_proxy.get_client_list(),
                                                                   batch_size=batch_size)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=True)

    def _construct_sign(self):
        self.keys_dict = dict()
        self.keys = list()
        sign_num = self.global_args.get('sign_num')
        if (None == sign_num):
            sign_num = 0
            logger.info("No client need to add watermark")
            for ind, (client_id, _) in enumerate(self.client_list.items()):
                self.keys_dict[client_id] = None
        else:
            logger.info(f"{sign_num} client(s) will inject watermark into their models")

            for i in range(self.global_args.get('client_num')):
                if i < self.global_args.get('sign_num'):
                    key = chain_proxy.construct_sign(self.global_args)
                    self.keys.append(key)
                else:
                    self.keys.append(None)
            for ind, (client_id, _) in enumerate(self.client_list.items()):
                self.keys_dict[client_id] = self.keys[ind]
            tmp_args = chain_proxy.construct_sign(self.global_args)
            self.model = ModelFactory().get_sign_model(model=self.global_args.get('model'),
                                                       class_num=self.global_args.get('class_num'),
                                                       in_channels=self.global_args.get('in_channels'),
                                                       watermark_args=tmp_args)
        return

    def _regist_client(self):
        for i in range(self.global_args['client_num']):
            chain_proxy.client_regist()
            print(f"----> 客户端 {i + 1} 注册完成，当前区块高度: {chain_proxy.get_block_height()}")
        self.client_list = chain_proxy.get_client_list()

    def _construct_client(self):
        # 客户端在初始化时，需要从服务器获取初始模型
        initial_model_state = self.server.download_model()
        if initial_model_state:
            self.model.load_state_dict(initial_model_state)

        for client_id, _ in self.client_list.items():
            new_client = self.client(client_id, self.train_dataloader_list[client_id], self.model,
                                     self.trainer, self.train_args, self.test_dataloader, self.keys_dict[client_id]
                                     , global_args=self.global_args)
            self.client_pool.append(new_client)

    # def run(self):
    #     print("========> 步骤1: 正在注册客户端到区块链...")
    #     self._regist_client()
    #     print("========> 步骤2: 正在构建数据加载器 (Dataloader)...")
    #     self._construct_dataloader()
    #     print("========> 步骤3: 正在构建签名/水印...")
    #     self._construct_sign()
    #     print("========> 步骤4: 正在构建客户端实例...")
    #     self._construct_client()
    #
    #     # --- 新增步骤：远程证明 (附带详细调试) ---
    #     print("\n========> DEBUG: 开始远程证明流程...")
    #     attestation_report = self.server.get_tee_attestation_report()
    #
    #     print(f"DEBUG: 从服务器获取的证明报告: {attestation_report}")
    #
    #     tee_public_key = None
    #
    #     if attestation_report and 'report_data' in attestation_report:
    #         report_data = attestation_report['report_data']
    #         print(f"DEBUG: 报告中的数据 (report_data): {report_data}")
    #
    #         expected_mrenclave = self.server.tee_aggregator.mrenclave
    #         print(f"DEBUG: 客户端期望的 MRENCLAVE: {expected_mrenclave}")
    #
    #         if report_data.get('mrenclave') == expected_mrenclave:
    #             print("----> DEBUG: MRENCLAVE 匹配成功!")
    #             tee_public_key = report_data.get('public_key')
    #             print(f"----> DEBUG: TEE公钥已赋值为: {tee_public_key}")
    #             print("----> 远程证明成功！所有客户端确认服务器TEE可信。\n")
    #         else:
    #             print("----> 远程证明失败！MRENCLAVE不匹配。")
    #             print(f"      收到的: {report_data.get('mrenclave')}")
    #             return
    #     else:
    #         print("----> 远程证明失败！报告格式不正确或为空。")
    #         return
    #
    #     # --- 远程证明结束 ---
    #
    #     print("========> 步骤5: 进入主训练循环...")
    #     for i in range(self.global_args['communication_round']):
    #         print(f"========> [第 {i + 1} 轮] 客户端训练并加密上传模型...")
    #         for client in self.client_pool:
    #             client.train(epoch=i)
    #
    #             client_model_state_dict = client.get_model_state_dict()
    #
    #             # 调试: 检查本轮循环中使用的公钥
    #             if tee_public_key is None:
    #                 print(f"FATAL ERROR: 在第 {i + 1} 轮加密前, tee_public_key 的值是 None!")
    #
    #             serializable_dict = OrderedDict({k: v.tolist() for k, v in client_model_state_dict.items()})
    #             model_payload = json.dumps(serializable_dict)
    #
    #             encrypted_package = {
    #                 "payload": model_payload,
    #                 "encrypted_with": tee_public_key
    #             }
    #
    #             self.server.upload_model(encrypted_package)
    #
    #             client.test(epoch=i)
    #             client.sign_test(epoch=i)
    #
    #         print(f"========> [第 {i + 1} 轮] 所有加密模型已上传，服务器已在TEE内完成解密和聚合。")
    #
    #         new_global_model_state = self.server.download_model()
    #
    #         if new_global_model_state:
    #             model_json = json.dumps(OrderedDict({k: v.tolist() for k, v in new_global_model_state.items()}))
    #             model_hash_hex = hashlib.sha256(model_json.encode('utf-8')).hexdigest()
    #             model_hash_int = int(model_hash_hex, 16)
    #             chain_proxy.upload_model_hash(model_hash_int, i + 1)
    #             print(f"----> 当前区块高度: {chain_proxy.get_block_height()}")
    #
    #         print(f"========> [第 {i + 1} 轮] 向客戶端分发新模型...")
    #         print(f"------------------------------------------------------------------")
    #         if new_global_model_state:
    #             for client in self.client_pool:
    #                 client.load_state_dict(new_global_model_state)
    #         else:
    #             print(f"WARNING: [第 {i + 1} 轮] 未能获取新的全局模型，客户端模型未更新。")
    # in task.py

    # in task.py

    def run(self):
        print("========> 步骤1: 正在注册客户端到区块链...")
        self._regist_client()
        print("========> 步骤2: 正在构建数据加载器 (Dataloader)...")
        self._construct_dataloader()
        print("========> 步骤3: 正在构建签名/水印...")
        self._construct_sign()
        print("========> 步骤4: 正在构建客户端实例...")
        self._construct_client()

        # --- 新增步骤：每个客户端独立进行远程证明 ---
        print("\n========> 新增步骤: 各个客户端独立验证服务器TEE的真实性 (防重放攻击)...")

        all_clients_verified = True
        tee_public_key = None  # 公钥只需获取一次，所有客户端获取到的应该都一样

        for client in self.client_pool:
            print(f"----> 客户端 {client.client_id} 开始进行远程证明...")

            # 1. 每个客户端生成自己独一无二的Nonce
            client_nonce = binascii.hexlify(os.urandom(16)).decode()
            print(f"      客户端 {client.client_id} 生成Nonce: {client_nonce}")

            # 2. 使用自己的Nonce请求证明
            attestation_report = self.server.get_tee_attestation_report(nonce=client_nonce)

            # 3. 独立验证收到的报告
            if attestation_report and \
                    attestation_report['report_data']['mrenclave'] == self.server.tee_aggregator.mrenclave and \
                    attestation_report['report_data']['nonce'] == client_nonce:

                print(f"----> 客户端 {client.client_id} 证明成功！")

                # 在第一次成功证明时，获取并存储公钥
                if tee_public_key is None:
                    tee_public_key = attestation_report['report_data']['public_key']
                    print(f"----> 已从首次证明中获取TEE公钥: {tee_public_key}")
            else:
                print(f"----> 客户端 {client.client_id} 证明失败！任务终止。")
                all_clients_verified = False
                break  # 任何一个客户端验证失败，则立刻中断整个流程

        # 如果有任何一个客户端验证失败，则不开始训练
        if not all_clients_verified:
            return

        print("\n----> 所有客户端均已成功验证TEE，联邦学习任务正式开始。")
        # --- 远程证明结束 ---

        print("========> 步骤5: 进入主训练循环...")
        for i in range(self.global_args['communication_round']):
            print(f"========> [第 {i + 1} 轮] 客户端训练并加密上传模型...")
            for client in self.client_pool:
                client.train(epoch=i)

                client_model_state_dict = client.get_model_state_dict()

                serializable_dict = OrderedDict({k: v.tolist() for k, v in client_model_state_dict.items()})
                model_payload = json.dumps(serializable_dict)

                encrypted_package = {
                    "payload": model_payload,
                    "encrypted_with": tee_public_key
                }

                BYZANTINE_CLIENT_NUM_F=2;
                AGGREGATION_STRATEGY = 'fedavg'
                self.server.upload_model(encrypted_package,byzantine_client_num=BYZANTINE_CLIENT_NUM_F,strategy=AGGREGATION_STRATEGY)

                client.test(epoch=i)
                client.sign_test(epoch=i)

            print(f"========> [第 {i + 1} 轮] 所有加密模型已上传，服务器已在TEE内完成解密和聚合。")

            new_global_model_state = self.server.download_model()

            if new_global_model_state:
                model_json = json.dumps(OrderedDict({k: v.tolist() for k, v in new_global_model_state.items()}))
                model_hash_hex = hashlib.sha256(model_json.encode('utf-8')).hexdigest()
                model_hash_int = int(model_hash_hex, 16)
                chain_proxy.upload_model_hash(model_hash_int, i + 1)
                print(f"----> 当前区块高度: {chain_proxy.get_block_height()}")

            print(f"========> [第 {i + 1} 轮] 向客戶端分发新模型...")
            print(f"------------------------------------------------------------------")
            if new_global_model_state:
                for client in self.client_pool:
                    client.load_state_dict(new_global_model_state)
            else:
                print(f"WARNING: [第 {i + 1} 轮] 未能获取新的全局模型，客户端模型未更新。")