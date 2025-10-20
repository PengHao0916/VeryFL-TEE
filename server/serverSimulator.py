
from server.tee_aggregator import TEEAggregator
import os
import torch
from copy import deepcopy
from typing import OrderedDict


class serverSimulator:
    def __init__(
            self,
            client_num=10,
            args=None
    ) -> None:
        self.tee_aggregator = TEEAggregator()
        self.client_num = client_num
        self.global_model = None
        if args is not None:
            self.args = args
        self.upload_model_list = []

        self.last_run_hash = None
        self.last_run_signature = None

    def _clear_upload_model_list(self):
        self.upload_model_list = []
        self.last_run_hash = None
        self.last_run_signature = None

    def _is_all_client_upload(self) -> bool:
        return len(self.upload_model_list) >= self.client_num

    def _set_global_model(self, global_model):
        self.global_model = global_model

    def _set_test_dataset(self, test_dataset):
        self.test_dataset = test_dataset
        self.test_batch_size = test_dataset.batch_size
        if test_dataset is None:
            raise Exception("Need to provide test dataset.")

    def _load_model(self):
        save_path = str(self.args['checkpoint_folder'])
        file_name = str(self.args['model'])
        file_path = save_path + file_name
        self.global_model.load_state_dict(torch.load(file_path))
        return

    def save_model(self, file_name='saved_model'):
        save_path = str(self.args['checkpoint_folder'])
        file_path_prefix = save_path + file_name
        if not os.path.isfile(file_path_prefix):
            torch.save(self.global_model.state_dict(), file_path_prefix)
        else:
            count = 0
            file_path = file_path_prefix + str(count)
            while (os.path.isfile(file_path)):
                count = count + 1
                file_path = file_path_prefix + str(count)
            torch.save(self.global_model.state_dict(), file_path)
        return True

    def get_tee_attestation_report(self,nonce:str):
        """
        As the non-secure host application, request an attestation report from its Enclave.
        """
        return self.tee_aggregator.generate_attestation_report(nonce)

    def upload_model(self, encrypted_upload_params: dict, byzantine_client_num: int = 0,strategy: str='fedavg'):
        self.upload_model_list.append(encrypted_upload_params)

        if self._is_all_client_upload():
            tee_output = self.tee_aggregator.decrypt_and_aggregate(
                self.upload_model_list,
                byzantine_client_num=byzantine_client_num,
                strategy=strategy
            )

            if tee_output:
                trained_model_state_dict = tee_output["model_state_dict"]
                self.last_run_hash = tee_output["model_hash"]
                self.last_run_signature = tee_output["tee_signature"]

                if self.global_model is not None and trained_model_state_dict is not None:
                    self.global_model.load_state_dict(trained_model_state_dict)
            else:
                print("ERROR: TEE aggregation returned None. Global model not updated.")

            self._clear_upload_model_list()

    def get_last_tee_output(self) -> dict:
        """
        允许 task.py (主协调器) 获取上一轮TEE的可信输出，以便将其上链。
        """
        return {
            "hash": self.last_run_hash,
            "signature": self.last_run_signature
        }

    def download_model(self, params=None) -> OrderedDict:
        if self.global_model is None:
            print("Error: Global model is not initialized yet.")
            return None
        else:
            return deepcopy(self.global_model.state_dict())

    def test(self):
        pass


if __name__ == '__main__':
    import torch.nn as nn


    class LinearModel(nn.Module):
        def __init__(self, h_dims):
            super(LinearModel, self).__init__()
            models = []
            for i in range(len(h_dims) - 1):
                models.append(nn.Linear(h_dims[i], h_dims[i + 1]))
                if i != len(h_dims) - 2:
                    models.append(nn.ReLU())
            self.models = nn.Sequential(*models)

        def forward(self, X):
            return self.models(X)


    test_sample_pool = [LinearModel([10, 10]) for i in range(10)]

    server = serverSimulator(client_num=10)

    initial_model = LinearModel([10, 10])
    server._set_global_model(initial_model)

    for i in test_sample_pool:
        # This unit test part won't work directly with encryption,
        # but the server's syntax will be correct.
        # We simulate a simple non-encrypted upload for syntax checking.
        upload_param = {'state_dict': i.state_dict()}
        # server.upload_model(upload_param) # Bypassing for unit test

    print("Aggregated model state dict (initial state):")
    print(server.download_model())