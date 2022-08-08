import json
import os
import re

import torch
from pykeen.datasets import FB15k237, FB15k, WN18

from models import TransE
from train_and_evaluate import TransETrain, Evaluation
from transe_boost import TransEBoost, TransEBoost2


class LoadMetaDataHandling:
    def __init__(self, folder_list, dataset_num_map):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.dataset_name = None
        self.num_entity = None
        self.num_relation = None
        self.folder_list = folder_list
        self.dataset_num_map = dataset_num_map
        # check if the necessary folders exist
        exist_dir = [dir_name for dir_name in os.listdir('./') if os.path.isdir('./' + dir_name) if
                     dir_name in self.folder_list]
        if len(exist_dir) != len(self.folder_list):
            dir_to_create = [dir_name for dir_name in self.folder_list if dir_name not in exist_dir]
            for direc in dir_to_create:
                os.mkdir(direc)

    def dataset_name_to_number(self, dataset_name):
        # if dataset_name == 'FB15k':
        #     automatic_input = '1'
        # elif dataset_name == 'FB15k237':
        #     automatic_input = '2'
        # elif dataset_name == 'WN18':
        #     automatic_input = '3'
        # return automatic_input
        return self.dataset_num_map[dataset_name]

    def resume_exp(self, folder):
        print('Experiment List: ')
        # print(folder)
        # print([dir_name for dir_name in os.listdir('./' + folder + '/')])
        for exp in [dir_name for dir_name in os.listdir('./' + folder + '/') if
                    os.path.isdir('./' + folder + '/' + dir_name)]:
            print(exp)
        select_exp_num = input('Choose a Exp. number from list: ')
        exp_dir_name = folder + '/' + 'exp_' + str(select_exp_num)
        with open(exp_dir_name + '/' + 'meta_data.json', 'r') as json_file:
            meta_data = json.load(json_file)
            json_file.close()
        automatic_input = self.dataset_name_to_number(dataset_name=meta_data['global']['dataset name'])
        self.get_dataset(automatic_input=automatic_input)
        transe_model = torch.load(exp_dir_name + '/' + 'transe_' + str(meta_data['global']['latest epoch']) + '.pt')
        return meta_data, exp_dir_name, transe_model

    def load(self, folder):
        choose_exp = input('Run a new experiment (y), or continue with an existing one (n) ?: ')
        if choose_exp == 'y':
            meta_data, exp_dir_name, transe_model = self.create_exp(folder=folder)
            torch.manual_seed(meta_data['global']['seed'])
            torch.cuda.manual_seed_all(meta_data['global']['seed'])
            return meta_data, exp_dir_name, transe_model, self.train_dataset, self.val_dataset, self.test_dataset
        elif choose_exp == 'n':
            meta_data, exp_dir_name, transe_model = self.resume_exp(folder=folder)
            torch.manual_seed(meta_data['global']['seed'])
            torch.cuda.manual_seed_all(meta_data['global']['seed'])
            return meta_data, exp_dir_name, transe_model, self.train_dataset, self.val_dataset, self.test_dataset

    def get_latest_exp(self, folder):
        exist_dir = [dir_name for dir_name in os.listdir('./' + folder + '/') if
                     os.path.isdir('./' + folder + '/' + dir_name)]
        if len(exist_dir) != 0:
            file_numbers = [int(re.findall('\d+', dir_name)[0]) for dir_name in exist_dir]
            if len(file_numbers) != 0:
                latest_exp_num = max(file_numbers)
            else:
                return 0
        else:
            return 0
        return latest_exp_num

    def create_exp(self, folder):
        new_exp_num = self.get_latest_exp(folder) + 1
        exp_dir_name = folder + '/' + 'exp_' + str(new_exp_num)
        os.mkdir(exp_dir_name)
        self.get_dataset()
        meta_data = {'global': {'device': input('Device: '),
                                'seed': int(input('Seed: ')),
                                'dataset name': self.dataset_name,
                                'num entity': self.num_entity,
                                'num relation': self.num_relation,
                                'emb dim': int(input('Embedding Dimension: ')),
                                'gamma': float(input('Gamma: ')),
                                'lr': float(input('Learning rate: ')),
                                'l2': float(input('L2 Weight Decay: ')),
                                'batch size': int(input('Batch Size: ')),
                                'latest epoch': 0,
                                'total epoch': int(input('total number of epochs: ')),
                                'description': input('Enter a brief description about this experiment: ')
                                },
                     'local': {}
                     }
        with open(exp_dir_name + '/' + 'meta_data.json', 'w+') as json_file:
            json.dump(meta_data, json_file, indent=4)
            json_file.close()
        # create a new transe model:
        print('Creating a new TransE model: ')
        transe_model = TransE(device=meta_data['global']['device'],
                              num_entity=meta_data['global']['num entity'],
                              num_relation=meta_data['global']['num relation'],
                              emb_dim=meta_data['global']['emb dim'],
                              gamma=meta_data['global']['gamma'],
                              seed=meta_data['global']['seed'])
        print('Done!!')
        return meta_data, exp_dir_name, transe_model

    def get_dataset(self, automatic_input=None):
        if automatic_input == None:
            select_dataset = input('''1)FB15K\n2)FB15K237\n3)WN18\n4)Exit\nEnter 1,2,3 or 4:''')
        else:
            select_dataset = automatic_input
        if select_dataset == '1':
            dataset = FB15k()
            self.dataset_name = 'FB15k'
        elif select_dataset == '2':
            dataset = FB15k237()
            self.dataset_name = 'FB15k237'
        elif select_dataset == '3':
            dataset = WN18()
            self.dataset_name = 'WN18'
        elif select_dataset == '4':
            return
        else:
            return
        # assign the values
        self.train_dataset = dataset.training.mapped_triples
        self.val_dataset = dataset.validation.mapped_triples
        self.test_dataset = dataset.testing.mapped_triples
        self.num_entity = dataset.num_entities.real
        self.num_relation = dataset.num_relations.real
        # print their shapes
        print('Training dataset size: ', self.train_dataset.shape)
        print('Validation dataset size: ', self.val_dataset.shape)
        print('Testing dataset size: ', self.test_dataset.shape)
        print('Number of Entities: ', self.num_entity)
        print('Number of relations: ', self.num_relation)
        print('Dataset Name: ', self.dataset_name)
        # return the values
        # return train_dataset, val_dataset, test_dataset, num_entity, num_relation, dataset_name

    def get_data(self):
        return self.train_dataset, self.val_dataset, self.test_dataset

    def meta_data_add_field(self, exp_dir_name, **kwargs):
        with open(exp_dir_name + '/' + 'meta_data.json', 'r+') as json_file:
            meta_data = json.load(json_file)
            for key, value in kwargs.items():
                meta_data['global'][str(key)] = value
            json_file.seek(0)
            json.dump(meta_data, json_file, indent=4)
            json_file.truncate()
            json_file.close()
            return meta_data


# # set transe model paratmeters:
# device = 'cuda'
# emb_dim = 50
# gamma = 1
#
# # set transe optimizer parameters:
# lr = 0.01
# weight_decay = 0
#
# # transe training parameters:
# batch_size = 36
# epoch = 50
#
# # set torch seeds:
# seed = 2022
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
#
# # TransEBoost parameters:
# end_epoch = 100


def main():
    folder_list = ['transe_original', 'transe_selftrain_type_1', 'transe_selftrain_type_2']
    dataset_num_map = {'FB15k': '1', 'FB15k237': '2', 'WN18': '3'}
    load_data = LoadMetaDataHandling(folder_list=folder_list, dataset_num_map=dataset_num_map)

    select_train_model = input('''
                               \n1) Train a basic TransE model? 
                               \n2) Evaluate a model? 
                               \n3) Training a self-training type 1 model?
                               \n4) Training a self-training type 2 model?
                               \n5) Exit 
                               \n Enter (1, 2, 3, 4, 5): ''')

    if select_train_model == '1':
        # Get the transe model:
        meta_data, exp_dir_name, transe_model, train_dataset, val_dataset, test_dataset = load_data.load(folder_list[0])
        # Create the optimizer:
        optimizer = torch.optim.SGD(transe_model.parameters(),
                                    lr=meta_data['global']['lr'],
                                    weight_decay=meta_data['global']['l2'])
        # Train the model:
        transe_model_train = TransETrain(train_dataset=train_dataset,
                                         batch_size=meta_data['global']['batch size'],
                                         num_entity=meta_data['global']['num entity'],
                                         model=transe_model,
                                         device=meta_data['global']['device'],
                                         optimizer=optimizer,
                                         start_epoch=meta_data['global']['latest epoch'] + 1,
                                         end_epoch=meta_data['global']['total epoch'],
                                         seed=meta_data['global']['seed'],
                                         folder=exp_dir_name)
        # train TransE model:
        transe_model_train.train()
        # save the model:
        # save_original_transe(model=transe_model,
        #                      folder='transe_neworg_models',
        #                      num_entity=num_entity,
        #                      num_relation=num_relation,
        #                      dataset_name=dataset_name,
        #                      mr=0,
        #                      mrr=0,
        #                      hits_at_10=0)

    elif select_train_model == '2':
        select_eva_model = input('''Select model to be evaluated:\n
                                  1) original model\n
                                  2) self training type 1\n
                                  3) self training type 2\n
                                  Enter (1,2,3):''')
        select_exp_num = input('Enter the exp. number: ')
        select_eva_model_num = list(map(int, input('''Enter range of models to be evaluated: ''').split()))
        if select_eva_model == '1':
            evaluate_model = folder_list[0] + '/'
        elif select_eva_model == '2':
            evaluate_model = folder_list[1] + '/'
        elif select_eva_model == '3':
            evaluate_model = folder_list[2] + '/'
        evaluate_model += 'exp_' + select_exp_num + '/'
        with open(evaluate_model + 'meta_data.json', 'r+') as json_file:
            meta_data = json.load(json_file)
            # get the appropriate dataset to evaluate the model
            automatic_input = load_data.dataset_name_to_number(dataset_name=meta_data['global']['dataset name'])
            load_data.get_dataset(automatic_input=automatic_input)
            train_dataset, val_dataset, test_dataset = load_data.get_data()
            for eva_model_num in range(select_eva_model_num[0], select_eva_model_num[1] + 1):
                eva_model_num = str(eva_model_num)
                # load the model to be evaluated
                print('Loading a TransE model from disk...')
                try:
                    transe_model = torch.load(evaluate_model + 'transe_' + eva_model_num + '.pt')
                except:
                    print('Error in loading model. Aborting.')
                    return -1
                print('Done!\nModel being evaluated: ', evaluate_model + 'transe_' + eva_model_num + '.pt')
                # specify the values for evaluation:
                eva_obj = Evaluation(data=val_dataset,
                                     model=transe_model,
                                     num_entity=meta_data['global']['num entity'],
                                     device=meta_data['global']['device'])
                # Evaluate TransE model:
                mr, mrr, hits_at_10 = eva_obj.evaluate_model()
                # store the values in meta data file for that model
                meta_data['local']['transe_' + eva_model_num]['MR'] = float(mr)
                meta_data['local']['transe_' + eva_model_num]['MRR'] = float(mrr)
                meta_data['local']['transe_' + eva_model_num]['Hits@10'] = float(hits_at_10)
            json_file.seek(0)
            json.dump(meta_data, json_file, indent=4)
            json_file.truncate()
            json_file.close()
        # save evaluation results:
        # save_evaluation_res(mr=mr,
        #                     mrr=mrr,
        #                     hits_at_10=hits_at_10,
        #                     dataset_name=dataset_name,
        #                     evaluate_model=evaluate_model)

    elif select_train_model == '3':
        meta_data, exp_dir_name, transe_model, train_dataset, val_dataset, test_dataset = load_data.load(folder_list[1])
        # Create the optimizer:
        optimizer = torch.optim.SGD(transe_model.parameters(),
                                    lr=meta_data['global']['lr'],
                                    weight_decay=meta_data['global']['l2'])
        # setup to train the model:
        self_train_type1 = TransEBoost(start_epoch=meta_data['global']['latest epoch'] + 1,
                                       end_epoch=meta_data['global']['total epoch'],
                                       train_data=train_dataset,
                                       seed=meta_data['global']['seed'],
                                       device=meta_data['global']['device'],
                                       batch_size=meta_data['global']['batch size'],
                                       model=transe_model,
                                       optimizer=optimizer,
                                       num_entity=meta_data['global']['num entity'],
                                       folder=exp_dir_name)
        # save the meta data:
        # save_boosted_meta(dataset_name=dataset_name,
        #                   num_entity=num_entity,
        #                   num_relation=num_relation,
        #                   start_epoch=start_epoch)
        # train the model:
        self_train_type1.train()

    elif select_train_model == '4':
        meta_data, exp_dir_name, transe_model, train_dataset, val_dataset, test_dataset = load_data.load(folder_list[2])
        # Check if transe_model is a dict or not:
        if str(type(transe_model)) != "<class 'dict'>":
            temp = transe_model
            transe_model = {'pre_model': temp,
                            'cur_model': temp
                            }
        if meta_data['global']['latest epoch'] == 0:
            num_model_train = int(input('Enter number of models to train: '))
            meta_data = load_data.meta_data_add_field(exp_dir_name=exp_dir_name,
                                                      cur_model_num=1,
                                                      num_model_train=num_model_train)
            start_epoch = 1
        else:
            start_epoch = meta_data['global']['latest epoch'] % meta_data['global']['total epoch']
            if start_epoch == 0:
                start_epoch = 1
            else:
                start_epoch += 1
        # Create the optimizer:
        optimizer = torch.optim.SGD(transe_model['cur_model'].parameters(),
                                    lr=meta_data['global']['lr'],
                                    weight_decay=meta_data['global']['l2'])
        # setup the transe boost 2
        boost2_obj = TransEBoost2(pre_model=transe_model['pre_model'],
                                  cur_model=transe_model['cur_model'],
                                  train_data=train_dataset,
                                  start_epoch=start_epoch,
                                  end_epoch=meta_data['global']['total epoch'],
                                  total_epoch=meta_data['global']['latest epoch'],
                                  seed=meta_data['global']['seed'],
                                  device=meta_data['global']['device'],
                                  batch_size=meta_data['global']['batch size'],
                                  start_model=meta_data['global']['cur_model_num'],
                                  end_model=meta_data['global']['num_model_train'],
                                  optimizer=optimizer,
                                  num_entity=meta_data['global']['num entity'],
                                  folder=exp_dir_name)
        # train:
        boost2_obj.train()

    elif select_train_model == '5':
        return 0


# def restore_boosted_model(num_entity, num_relation):
#     file_list = os.listdir('transe_boosted_models/')
#     if len(file_list) == 0:
#         print('Creating a new TransE model: ')
#         transe_model = TransE(device=device,
#                               num_entity=num_entity,
#                               num_relation=num_relation,
#                               emb_dim=emb_dim,
#                               gamma=gamma,
#                               seed=seed)
#         print('Done!!')
#         start_epoch = 1
#     else:
#         print('Loading last trained model...')
#         file_numbers = [int(re.findall('\d+', i)[0]) for i in file_list]
#         model_num = max(file_numbers)
#         try:
#             transe_model = torch.load('transe_boosted_models/transe_boost_model_' + str(model_num) + '.pt')
#         except:
#             print('Error encountered while loading model. Aborting.')
#             return -1, -1
#         start_epoch = model_num + 1
#         print('Done!')
#     return transe_model, start_epoch
#
#
# def save_original_transe(model, folder, num_entity, num_relation, dataset_name, mr, mrr, hits_at_10):
#     file_list = os.listdir(folder + '/')
#     meta_data = {'Device': device,
#                  'Seed': seed,
#                  'Dataset Name': dataset_name,
#                  'Number of Entities': num_entity,
#                  'Number of Relations': num_relation,
#                  'Embedding Dimension': emb_dim,
#                  'Gamma': gamma,
#                  'Learning Rate': lr,
#                  'L2': weight_decay,
#                  'Model Number': 0,
#                  'Epochs': epoch,
#                  'Batch Size': batch_size,
#                  'MR': float(mr),
#                  'MRR': float(mrr),
#                  'Hits@10': float(hits_at_10)}
#     if len(file_list) == 0:
#         meta_data['Model Number'] = 1
#     else:
#         file_numbers = [int(re.findall('\d+', i)[0]) for i in file_list]
#         model_num = max(file_numbers)
#         model_num += 1
#         meta_data['Model Number'] = model_num
#     torch.save(model, folder + '/transe_org_model_' + str(meta_data['Model Number']) + '.pt')
#     with open(folder + '/meta_data_' + str(meta_data['Model Number']) + '.json', 'w') as json_file:
#         json.dump(meta_data, json_file, indent=4)
#         json_file.close()
#     print('Saving done !!')
#
#
# def load_original_transe(num_entity, num_relation):
#     file_list = os.listdir('transe_conorg_models/')
#     if len(file_list) == 0:
#         print('No existing model found')
#         print('Creating a new TransE model: ')
#         transe_model = TransE(device=device,
#                               num_entity=num_entity,
#                               num_relation=num_relation,
#                               emb_dim=emb_dim,
#                               gamma=gamma,
#                               seed=seed)
#         print('Done!!')
#     else:
#         file_numbers = [int(re.findall('\d+', i)[0]) for i in file_list]
#         model_num = max(file_numbers)
#         try:
#             transe_model = torch.load('transe_conorg_models/transe_org_model_' + str(model_num) + '.pt')
#         except:
#             print('Error encountered while loading model. Aborting.')
#             return -1
#     return transe_model
#
#
# def save_evaluation_res(mr, mrr, hits_at_10, dataset_name, evaluate_model):
#     meta_data = {'Device': device,
#                  'Seed': seed,
#                  'Dataset Name': dataset_name,
#                  'Model Name': evaluate_model.replace('/', '-'),
#                  'MR': float(mr),
#                  'MRR': float(mrr),
#                  'Hits@10': float(hits_at_10)}
#     with open('evaluation_results/meta_data_' + meta_data['Model Name'][:-3] + '.json', 'w') as json_file:
#         json.dump(meta_data, json_file, indent=4)
#         json_file.close()
#     print('Saving done !!')
#
#
# def save_boosted_meta(dataset_name, num_entity, num_relation, start_epoch):
#     meta_data = {'Device': device,
#                  'Seed': seed,
#                  'Dataset Name': dataset_name,
#                  'Number of Entities': num_entity,
#                  'Number of Relations': num_relation,
#                  'Embedding Dimension': emb_dim,
#                  'Gamma': gamma,
#                  'Learning Rate': lr,
#                  'L2': weight_decay,
#                  'Start Epoch': start_epoch,
#                  'Batch Size': batch_size}
#     with open('transe_boosted_models/meta_data' + str(meta_data['Start Epoch']) + '.json', 'w') as json_file:
#         json.dump(meta_data, json_file, indent=4)
#         json_file.close()
#     print('Saving Done!!')


if __name__ == '__main__':
    main()
