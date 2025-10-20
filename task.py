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

        # --- 远程证明 (保持不变) ---
        print("\n========> 步骤: 各个客户端独立验证服务器TEE的真实性 (防重放攻击)...")

        all_clients_verified = True
        tee_public_key = None  # 公钥只需获取一次

        for client in self.client_pool:
            print(f"----> 客户端 {client.client_id} 开始进行远程证明...")

            client_nonce = binascii.hexlify(os.urandom(16)).decode()
            print(f"      客户端 {client.client_id} 生成Nonce: {client_nonce}")

            attestation_report = self.server.get_tee_attestation_report(nonce=client_nonce)

            if attestation_report and \
                    attestation_report['report_data']['mrenclave'] == self.server.tee_aggregator.mrenclave and \
                    attestation_report['report_data']['nonce'] == client_nonce:

                print(f"----> 客户端 {client.client_id} 证明成功！")

                if tee_public_key is None:
                    tee_public_key = attestation_report['report_data']['public_key']
                    print(f"----> 已从首次证明中获取TEE公钥: {tee_public_key}")
            else:
                print(f"----> 客户端 {client.client_id} 证明失败！任务终止。")
                all_clients_verified = False
                break

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

                BYZANTINE_CLIENT_NUM_F = 2;
                # --- !! 关键 !! ---
                # 您可以在这里切换策略，例如 'trimmed_mean'
                AGGREGATION_STRATEGY = 'trimmed_mean'
                # --- 结束 ---
                self.server.upload_model(encrypted_package, byzantine_client_num=BYZANTINE_CLIENT_NUM_F,
                                         strategy=AGGREGATION_STRATEGY)

                client.test(epoch=i)
                client.sign_test(epoch=i)

            print(f"========> [第 {i + 1} 轮] 所有加密模型已上传，服务器已在TEE内完成聚合、哈希和签名。")

            # --- 关键修改：用 TEE 证明替换旧的哈希逻辑 ---

            # 1. 从服务器获取TEE的输出：模型 + 证明
            new_global_model_state = self.server.download_model()
            tee_proof = self.server.get_last_tee_output()

            hash_from_tee = tee_proof["hash"]
            sig_from_tee = tee_proof["signature"]

            # 2. 检查 TEE 是否成功运行
            if not new_global_model_state or not hash_from_tee:
                print(f"WARNING: [第 {i + 1} 轮] TEE未能成功聚合或生成证明，跳过本轮更新。")
                continue  # 跳到下一轮循环

            # 3. 将 TEE 的可信证明上传到区块链
            #    我们调用新的函数 upload_model_proof
            chain_proxy.upload_model_proof(hash_from_tee, sig_from_tee, i + 1)
            print(f"----> 当前区块高度: {chain_proxy.get_block_height()}")

            # 4. 客户端验证并分发模型 (堵上漏洞的关键)
            print(f"========> [第 {i + 1} 轮] 客户端开始验证模型完整性和TEE签名...")

            # 4a. 客户端本地计算收到的模型的哈希
            model_json = json.dumps(OrderedDict({k: v.tolist() for k, v in new_global_model_state.items()}))
            local_hash = hashlib.sha256(model_json.encode('utf-8')).hexdigest()

            # 4b. 从区块链获取可信的哈希和签名
            #    (在真实应用中，这里会调用 chain_proxy.get_proof(i+1) 来查询)
            #    (在我们的模拟中，我们直接使用刚从TEE获取的值，因为我们信任 chain_proxy)
            hash_from_chain = hash_from_tee
            sig_from_chain = sig_from_tee

            # 4c. 客户端本地模拟验证签名
            #    (真实应用中，客户端会用 TEE 公钥执行 `public_key.verify(signature, hash)` )
            #    (模拟中，我们用 "已知" 的私钥重新计算预期签名并对比)
            tee_private_key_for_sim = self.server.tee_aggregator._private_key
            expected_sig_payload = f"{hash_from_chain}{tee_private_key_for_sim}"
            expected_signature = hashlib.sha256(expected_sig_payload.encode('utf-8')).hexdigest()

            # 4d. 执行双重验证
            if local_hash != hash_from_chain:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"FATAL ERROR: [第 {i + 1} 轮] 验证失败！模型完整性被破坏！")
                print(f"             (服务器在TEE之外篡改了模型)")
                print(f"  客户端本地哈希: {local_hash}")
                print(f"  区块链可信哈希: {hash_from_chain}")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("任务终止。")
                break  # 严重安全漏洞，终止训练

            elif sig_from_chain != expected_signature:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"FATAL ERROR: [第 {i + 1} 轮] 验证失败！TEE签名无效！")
                print(f"             (服务器伪造了链上的哈希和签名)")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("任务终止。")
                break  # 严重安全漏洞，终止训练

            else:
                print("----> 验证成功！模型完整性 和 TEE来源 已确认。")
                print(f"========> [第 {i + 1} 轮] 向客戶端分发新模型...")
                print(f"------------------------------------------------------------------")
                for client in self.client_pool:
                    client.load_state_dict(new_global_model_state)

            # --- 修改结束 ---