# server/tee_aggregator.py

import torch
from typing import List, Dict
from collections import OrderedDict
import hashlib
import json
import numpy as np  # 引入numpy用于计算


class TEEAggregator:
    def __init__(self):
        print("INFO: Simulated TEE Aggregator Initialized.")
        # MRENCLAVE的计算方式仅为示例
        self.mrenclave = self._calculate_mrenclave()

        self.public_key = "simulated_enclave_public_key_fedavg_only"
        self._private_key = "simulated_enclave_private_key_fedavg_only"
        print(f"INFO: TEE MRENCLAVE is: {self.mrenclave}")

    def _calculate_mrenclave(self):
        """动态计算MRENCLAVE，确保代码变更后哈希值也更新"""
        code_str = ""
        # 将所有聚合函数的代码都纳入哈希计算
        code_str += self._fedavg_aggregate.__code__.co_code.hex()
        code_str += self._trimmed_mean_aggregate.__code__.co_code.hex()  # <-- 新增
        return hashlib.sha256(code_str.encode()).hexdigest()

    def generate_attestation_report(self, nonce: str):
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

    def _fedavg_aggregate(self, models: List[OrderedDict]) -> OrderedDict:
        """
        在Enclave内部，对所有模型执行FedAvg（简单平均）。
        """
        if not models:
            return OrderedDict()

        aggregated_model = OrderedDict()
        model_keys = models[0].keys()

        for key in model_keys:
            sum_of_params = torch.stack([model[key].float() for model in models], dim=0).sum(dim=0)
            avg_param = sum_of_params / len(models)
            aggregated_model[key] = avg_param

        return aggregated_model

    def _trimmed_mean_aggregate(self, models: List[OrderedDict], byzantine_client_num: int) -> OrderedDict:
        """
        在Enclave内部执行Trimmed Mean算法。
        """
        print(f"INFO: TEE entering Trimmed Mean aggregation. f={byzantine_client_num}")
        num_clients = len(models)

        # 边缘情况处理：如果恶意客户端数量大于等于总数的一半，算法失效
        if byzantine_client_num * 2 >= num_clients:
            print(
                f"WARNING: Trimmed Mean requires 2*f < n. (f={byzantine_client_num}, n={num_clients}). Falling back to FedAvg.")
            return self._fedavg_aggregate(models)

        # 1. 将模型参数字典转换为向量列表，便于计算距离
        model_vectors = []
        for model_dict in models:
            # 将所有参数层展平并拼接成一个长向量
            vec = torch.cat([param.view(-1) for param in model_dict.values()])
            model_vectors.append(vec)

        # 2. 计算每个模型到其他所有模型的距离平方之和，作为分数
        scores = []
        for i in range(num_clients):
            current_score = 0
            for j in range(num_clients):
                if i == j:
                    continue
                # 计算欧氏距离的平方
                dist_sq = torch.sum((model_vectors[i] - model_vectors[j]) ** 2)
                current_score += dist_sq
            scores.append(current_score)

        # 3. 找出分数最高的f个客户端的索引（这些是“离群”的，最可疑）
        #    我们对分数进行排序，并获取原始索引
        sorted_indices = sorted(range(num_clients), key=lambda k: scores[k], reverse=True)

        # 4. 确定要剔除的客户端索引
        indices_to_trim = sorted_indices[:byzantine_client_num]
        print(f"INFO: Trimming clients with indices: {indices_to_trim}")

        # 5. 构建一个只包含“诚实”客户端模型的新列表
        trusted_models = []
        for i in range(num_clients):
            if i not in indices_to_trim:
                trusted_models.append(models[i])

        print(f"INFO: Aggregating with {len(trusted_models)} trusted models.")
        # 6. 对筛选后的“诚实”客户端模型列表执行标准的FedAvg
        return self._fedavg_aggregate(trusted_models)

    def decrypt_and_aggregate(self, encrypted_models: List[Dict], byzantine_client_num: int,
                              strategy: str = 'trimmed_mean'):
        """
        TEE核心入口: 解密后，根据策略选择聚合算法。
        """
        print(f"INFO: Entering TEE for secure aggregation using '{strategy}' strategy...")

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

        # --- 聚合阶段 (根据策略选择) ---
        aggregated_result = None
        if strategy == 'trimmed_mean':
            aggregated_result = self._trimmed_mean_aggregate(decrypted_model_list, byzantine_client_num)
        elif strategy == 'fedavg':
            aggregated_result = self._fedavg_aggregate(decrypted_model_list)
        # 您可以继续添加 'krum' 等其他策略
        else:
            print(f"WARNING: Unknown aggregation strategy '{strategy}'. Defaulting to FedAvg.")
            aggregated_result = self._fedavg_aggregate(decrypted_model_list)

        print("INFO: Secure aggregation complete. Exiting TEE.")
        return aggregated_result