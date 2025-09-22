# #This implement the base class of server
# from server.base.baseAggregator import ServerAggregator
# import os
# import torch
# from copy import deepcopy
# from typing import OrderedDict
#
# class serverSimulator:
#     # A local serverSimulator to simulate the blockchain
#     # This serverSimulator act as the same with the serverProxy in chainfl.
#
#     def __init__(
#         self,
#         aggergator:ServerAggregator,
#         client_num=10,
#         args = None
#         #args params give the server some other information to initialize
#         #args is a dict which can contain below keys:
#         # 1.checkpoint_folder: indicate the saved model path
#         # 2.model: indicate the saved model name
#         # 3.train_method: indicate the aggregated method used,
#         #                 attention fedavg is used if provided method doesn't implemented
#         #
#     ) -> None:
#         self.aggergator = aggergator
#         self.client_num = client_num
#         self.global_model = None
#         if args != None:
#             self.args = args
#         self.upload_model_list = []
#     def _clear_upload_model_list(self):
#         self.upload_model_list = []
#
#     def _is_all_client_upload(self) -> bool:
#         return len(self.upload_model_list) >= self.client_num
#
#     def _set_global_model(self, global_model):
#         self.global_model = global_model
#
#     def _set_test_dataset(self, test_dataset):
#         self.test_dataset = test_dataset
#         self.test_batch_size = test_dataset.batch_size
#         if test_dataset == None:
#             raise Exception("Need to provide test dataset.")
#
#     def _load_model(self):
#         save_path = str(self.args['checkpoint_folder'])
#         file_name = str(self.args['model'])
#         file_path = save_path + file_name
#         self.global_model.load_state_dict(torch.load(file_path))
#         return
#
#     def save_model(self,file_name='saved_model'):
#         save_path = str(self.args['checkpoint_folder'])
#         #run after the global model was update
#         #save the torch state_dict
#         file_path_prefix = save_path + file_name
#         if not os.path.isfile(file_path_prefix):
#             torch.save(self.global_model.state_dict(),file_path_prefix)
#         else:
#             count = 0
#             file_path = file_path_prefix + str(count)
#             while(os.path.isfile(file_path)):
#                 count = count + 1
#                 file_path = file_path_prefix + str(count)
#             torch.save(self.global_model.state_dict(), file_path)
#         return True
#
#     def upload_model(self,upload_params:dict):
#         model_state_dict = upload_params['state_dict']
#         self.upload_model_list.append(model_state_dict)
#         if(self._is_all_client_upload()):
#             trained_model = self.aggergator.aggregate(self.upload_model_list)
#             self._set_global_model(trained_model)
#             self._clear_upload_model_list()
#
#     def download_model(self,params=None) -> OrderedDict:
#         if(self.global_model == None): return "Failed to get global model"
#         else : return deepcopy(self.global_model)
#
#     def test():
#         pass
#
# if __name__ == '__main__':
#     import torch.nn as nn
#     class LinearModel(nn.Module):
#         def __init__(self, h_dims):
#             super(LinearModel,self).__init__()
#
#             models = []
#             for i in range(len(h_dims) - 1):
#                 models.append(nn.Linear(h_dims[i], h_dims[i + 1]))
#                 if i != len(h_dims) - 2:
#                     models.append(nn.ReLU())
#             self.models = nn.Sequential(*models)
#         def forward(self, X):
#             return self.models(X)
#
#     test_sample_pool = [LinearModel(10) for i in range(10)]
#     a = ServerAggregator()
#     server = serverSimulator(a)
#     for i in test_sample_pool:
#         upload_param = {'state_dict': i.state_dict()}
#         server.upload_model(upload_param)
# This implement the base class of server
# from server.base.baseAggregator import ServerAggregator # <-- 不再需要
# server/serverSimulator.py

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

    def _clear_upload_model_list(self):
        self.upload_model_list = []

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

    # def upload_model(self, encrypted_upload_params: dict):
    #     self.upload_model_list.append(encrypted_upload_params)
    #
    #     if self._is_all_client_upload():
    #         trained_model_state_dict = self.tee_aggregator.decrypt_and_aggregate(self.upload_model_list)
    #
    #         if self.global_model is not None and trained_model_state_dict is not None:
    #             self.global_model.load_state_dict(trained_model_state_dict)
    #
    #         self._clear_upload_model_list()
    def upload_model(self, encrypted_upload_params: dict, byzantine_client_num: int = 0):
        # 服务器现在接收的是加密后的参数
        self.upload_model_list.append(encrypted_upload_params)

        if self._is_all_client_upload():
            # 将整个加密列表和恶意客户端数量f，一同传递给TEE
            trained_model_state_dict = self.tee_aggregator.decrypt_and_aggregate(
                self.upload_model_list,
                byzantine_client_num=byzantine_client_num
            )

            if self.global_model is not None and trained_model_state_dict is not None:
                self.global_model.load_state_dict(trained_model_state_dict)

            self._clear_upload_model_list()

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