# server/tee_aggregator.py

import torch
from typing import List, Dict
from collections import OrderedDict
import hashlib
import json  # <-- 导入json
import numpy as np

class TEEAggregator:
    def __init__(self):
        print("INFO: Simulated TEE Aggregator Initialized.")
        enclave_code = self._internal_aggregate.__code__.co_code
        self.mrenclave = hashlib.sha256(enclave_code).hexdigest()

        self.public_key = "simulated_enclave_public_key_12345"
        self._private_key = "simulated_enclave_private_key_abcde"
        print(f"INFO: TEE MRENCLAVE is: {self.mrenclave}")

    def generate_attestation_report(self,nonce:str):
        print("INFO: Simulated TEE is generating an attestation report...")
        report = {
            "mrenclave": self.mrenclave,
            "public_key": self.public_key,
            "nonce": nonce,
            "status": "OK"
        }
        signed_report = {
            "report_data": report,
            "signature": "simulated_intel_hardware_signature"
        }
        return signed_report

    def _krum_select(self, models: List[OrderedDict], f: int) -> List[OrderedDict]:
        """
        在Enclave内部运行Krum算法，筛选出诚实的模型。
        f: 假设的拜占庭（恶意）客户端的最大数量。
        """
        print(f"INFO: TEE entering Krum selection. f={f}")

        # 将模型字典转换为向量，以便计算距离
        model_vectors = []
        for model in models:
            vec = torch.cat([param.view(-1) for param in model.values()])
            model_vectors.append(vec)

        n = len(model_vectors)
        if n <= 2 * f + 2:
            print("WARNING: Not enough clients for Krum selection. Skipping.")
            return models

    def _median_aggregate(self, models: List[OrderedDict]) -> OrderedDict:
            """
            在Enclave内部，对每个参数执行坐标中位数聚合。
            """
            print("INFO: TEE entering Coordinate-wise Median aggregation.")

            if not models:
                return None

            aggregated_model = OrderedDict()
            # 获取第一个模型的键，作为所有模型参数的参考
            model_keys = models[0].keys()

            for key in model_keys:
                # 1. 将所有客户端在同一个参数key上的张量堆叠起来
                #    例如，对于'layer1.weight'，我们得到一个形状为 [num_clients, a, b, c] 的大张量
                stacked_tensors = torch.stack([model[key] for model in models])

                # 2. 沿着客户端维度（dim=0）计算中位数
                #    torch.median会返回两个值：中位数的值和对应的索引，我们只需要第一个
                median_tensor = torch.median(stacked_tensors, dim=0).values

                aggregated_model[key] = median_tensor

            print("INFO: Coordinate-wise Median aggregation complete.")
            return aggregated_model

    def _internal_aggregate(self, raw_client_model_list: List[OrderedDict]) -> OrderedDict:
        # 计算两两之间的欧氏距离平方
        distances = torch.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(model_vectors[i] - model_vectors[j]) ** 2
                distances[i, j] = dist
                distances[j, i] = dist

        # 为每个模型计算分数：到最近的 n-f-1 个邻居的距离之和
        scores = torch.zeros(n)
        for i in range(n):
            sorted_dists, _ = torch.sort(distances[i])
            # 第0个是自己，所以从1开始取 n-f-1 个
            scores[i] = torch.sum(sorted_dists[1: n - f])

        # 选出分数最低的 n-f 个模型作为诚实模型
        _, top_indices = torch.topk(scores, n - f, largest=False)

        selected_models = [models[i] for i in top_indices]
        print(f"INFO: Krum selected {len(selected_models)} models out of {n}.")

        return selected_models

    def _internal_aggregate(self, raw_client_model_list: List[OrderedDict]) -> OrderedDict:
        """
        这是Enclave内部的、私有的聚合逻辑。它不应该被外部直接调用。
        """
        if not raw_client_model_list:
            return None

        if not all(isinstance(model, dict) for model in raw_client_model_list):
            return None

        aggregated_model = OrderedDict()
        first_model_keys = raw_client_model_list[0].keys()

        for key in first_model_keys:
            # 将列表中的张量转换为正确的dtype以进行stack
            tensors = [model[key].float() for model in raw_client_model_list]
            sum_of_weights = torch.stack(tensors).sum(0)
            aggregated_model[key] = sum_of_weights / len(raw_client_model_list)

        return aggregated_model

    # in server/tee_aggregator.py

    # ... (其他代码保持不变) ...

    def decrypt_and_aggregate(self, encrypted_models: List[Dict], byzantine_client_num: int):  # <-- 移除了默认值
        """
        TEE核心入口: 解密后，执行一个默认开启的两阶段安全协议。
        1. Krum筛选 -> 2. 中位数聚合
        byzantine_client_num: 预设的最大恶意客户端数量(f)，用于Krum筛选。
        """
        print("INFO: Entering TEE for two-stage secure aggregation...")

        # --- 解密阶段 (保持不变) ---
        decrypted_model_list = []
        for encrypted_model in encrypted_models:
            if encrypted_model.get("encrypted_with") != self.public_key:
                print(f"ERROR: TEE received data encrypted with wrong key!")
                return None

            try:
                model_json = encrypted_model["payload"]
                model_state_dict_list = json.loads(model_json)

                model_state_dict = OrderedDict()
                for key, value in model_state_dict_list.items():
                    model_state_dict[key] = torch.tensor(value)

                decrypted_model_list.append(model_state_dict)
            except Exception as e:
                print(f"ERROR: TEE failed to decrypt or decode payload. Error: {e}")
                return None

        print(f"INFO: TEE successfully decrypted {len(decrypted_model_list)} models.")

        # --- 两阶段安全协议 ---
        # 阶段一: 使用Krum进行异常模型筛选
        honest_models = self._krum_select(decrypted_model_list, f=byzantine_client_num)

        # 阶段二: 对筛选后的诚实模型，使用坐标中位数进行稳健聚合
        aggregated_result = self._median_aggregate(honest_models)
        # --- 协议结束 ---

        print("INFO: Two-stage secure aggregation complete. Exiting TEE.")
        return aggregated_result