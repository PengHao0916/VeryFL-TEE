# server/tee_aggregator.py

import torch
from typing import List, Dict
from collections import OrderedDict
import hashlib
import json


class TEEAggregator:
    def __init__(self):
        print("INFO: Simulated TEE Aggregator Initialized for FedAvg.")
        # MRENCLAVE的计算方式仅为示例
        enclave_code = self._fedavg_aggregate.__code__.co_code
        self.mrenclave = hashlib.sha256(enclave_code).hexdigest()

        self.public_key = "simulated_enclave_public_key_fedavg_only"
        self._private_key = "simulated_enclave_private_key_fedavg_only"
        print(f"INFO: TEE MRENCLAVE is: {self.mrenclave}")

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
        print("INFO: TEE entering FedAvg aggregation.")
        if not models:
            return OrderedDict()

        # 初始化一个空的 state_dict 用于存储聚合结果
        aggregated_model = OrderedDict()

        # 获取第一个模型的键，作为所有模型参数的参考
        model_keys = models[0].keys()

        for key in model_keys:
            # 将所有客户端在同一个参数key上的张量堆叠起来，然后计算平均值
            sum_of_params = torch.stack([model[key].float() for model in models], dim=0).sum(dim=0)
            avg_param = sum_of_params / len(models)
            aggregated_model[key] = avg_param

        print("INFO: FedAvg aggregation complete.")
        return aggregated_model

    def decrypt_and_aggregate(self, encrypted_models: List[Dict], byzantine_client_num: int):
        """
        TEE核心入口: 解密后，直接执行FedAvg聚合。
        """
        print("INFO: Entering TEE for secure aggregation (FedAvg only)...")

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

        # --- 聚合阶段 (直接FedAvg) ---
        aggregated_result = self._fedavg_aggregate(decrypted_model_list)
        # --- 协议结束 ---

        print("INFO: Secure aggregation complete. Exiting TEE.")
        return aggregated_result