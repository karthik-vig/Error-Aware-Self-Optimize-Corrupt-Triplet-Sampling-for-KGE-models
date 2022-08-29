import json
import os
import re

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pykeen.datasets import FB15k237, FB15k, WN18

from models import TransE

class SaveData:
    def __init__(self, folder):
        self.folder = folder

    def save(self, model, epoch, avg_loss, **kwargs):
        file_name = 'transe_' + str(epoch) + '.pt'
        file_path = self.folder + '/'
        try:
            torch.save(model, file_path + file_name)
            with open(file_path + 'meta_data.json', 'r+') as json_file:
                meta_data = json.load(json_file)
                meta_data['local'][file_name[:-3]] = {'Average Training Loss': float(avg_loss),
                                                 'MR': -1,
                                                 'MRR': -1,
                                                 'Hits@10': -1}
                meta_data['global']['latest epoch'] = int(epoch)
                for key, value in kwargs.items():
                    meta_data['global'][key] = value
                json_file.seek(0)
                json.dump(meta_data, json_file, indent=4)
                json_file.truncate()
                json_file.close()
        except:
            print('Save failed.')


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
        self.check_folder()

    def check_folder(self):
        # check if the necessary folders exist
        exist_dir = [dir_name for dir_name in os.listdir('./') if os.path.isdir('./' + dir_name) if
                     dir_name in self.folder_list]
        if len(exist_dir) != len(self.folder_list):
            dir_to_create = [dir_name for dir_name in self.folder_list if dir_name not in exist_dir]
            for direc in dir_to_create:
                os.mkdir(direc)

    def dataset_name_to_number(self, dataset_name):
        return self.dataset_num_map[dataset_name]

    def select_folder(self):
        print('Select model folder:')
        for index, folder in enumerate(self.folder_list):
            print(str(index+1) + ') ' + folder)
        select_eva_model = int(input('Enter a number: '))
        if select_eva_model > 0 and select_eva_model <= len(self.folder_list):
            return self.folder_list[select_eva_model - 1]
        else:
            return -1

    def select_exp(self, folder):
        print('Experiment List: ')
        for exp in [dir_name for dir_name in os.listdir('./' + folder + '/') if
                    os.path.isdir('./' + folder + '/' + dir_name)]:
            print(exp)
        select_exp_num = input('Choose a Exp. number from list: ')
        return select_exp_num

    def select_model_num(self, exp_dir_name, range_mode=True):
        model_num_list = [int(re.findall('\d+', model_name)[0]) for model_name in os.listdir('./' + exp_dir_name) if '.pt' in model_name]
        start_model_num = min(model_num_list)
        end_model_num = max(model_num_list)
        print('Model number range is: ', start_model_num, ' to ', end_model_num)
        if range_mode:
            select_model_num = list(map(int, input('Enter a model number range: ').split()))
        else:
            select_model_num = int(input('Enter a model number: '))
        return select_model_num

    def resume_exp(self, folder):
        select_exp_num = self.select_exp(folder=folder)
        exp_dir_name = folder + '/' + 'exp_' + str(select_exp_num)
        with open(exp_dir_name + '/' + 'meta_data.json', 'r') as json_file:
            meta_data = json.load(json_file)
            json_file.close()
        automatic_input = self.dataset_name_to_number(dataset_name=meta_data['global']['dataset name'])
        self.choose_dataset(automatic_input=automatic_input)
        transe_model = torch.load(exp_dir_name + '/' + 'transe_' + str(meta_data['global']['latest epoch']) + '.pt')
        return meta_data, exp_dir_name, transe_model

    def load(self, folder):
        choose_exp = input('Run a new experiment (y), or continue with an existing one (n) ?: ')
        if choose_exp == 'y':
            meta_data, exp_dir_name, transe_model = self.create_exp(folder=folder)
        else:
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
        self.choose_dataset()
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
                                'start model': None,
                                'description': input('Enter a brief description about this experiment: ')
                                },
                     'local': {}
                     }
        # Choose the source of transe model to start training from:
        transe_model_select = input('Create a new model (y) or use a existing model (n): ')
        if transe_model_select == 'y':
            # create a new transe model:
            print('Creating a new TransE model: ')
            transe_model = {'cur_model': TransE(device=meta_data['global']['device'],
                                                num_entity=meta_data['global']['num entity'],
                                                num_relation=meta_data['global']['num relation'],
                                                emb_dim=meta_data['global']['emb dim'],
                                                gamma=meta_data['global']['gamma'],
                                                seed=meta_data['global']['seed'])
                            }
            print('Done!!')
            # Set the start model choice in meta-data:
            meta_data['global']['start model'] = 'new TransE model'
        else:
            transe_model, model_path = self.get_model()
            # Set the start model choice in meta-data:
            meta_data['global']['start model'] = model_path
        # Save the meta-data to disk:
        with open(exp_dir_name + '/' + 'meta_data.json', 'w+') as json_file:
            json.dump(meta_data, json_file, indent=4)
            json_file.close()
        return meta_data, exp_dir_name, transe_model

    def get_model(self):
        print('Select a folder: ')
        for index, folder in enumerate(self.folder_list):
            print(str(index + 1) + ') ' + folder)
        folder_choice = input('Enter a number: ')
        exp_choice = input('Enter a exp. number: ')
        model_choice = input('Enter a model number: ')
        folder = self.folder_list[int(folder_choice) - 1]
        model_path = folder + '/' + 'exp_' + exp_choice + '/' + 'transe_' + model_choice + '.pt'
        transe_model = torch.load(model_path)
        return transe_model, model_path

    def choose_dataset(self, automatic_input=None):
        if automatic_input == None:
            select_dataset = input('''1)FB15K\n2)FB15K237\n3)WN18\n4)Exit (any other input will lead to exit)\nEnter 1,2,3 or 4:''')
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
        else:
            return
        # assign the values
        self.train_dataset = dataset.training.mapped_triples
        self.val_dataset = dataset.validation.mapped_triples
        self.test_dataset = dataset.testing.mapped_triples
        self.num_entity = dataset.num_entities.real
        self.num_relation = dataset.num_relations.real
        # print their shapes
        print('Dataset Name: ', self.dataset_name)
        print('Training dataset size: ', self.train_dataset.shape)
        print('Validation dataset size: ', self.val_dataset.shape)
        print('Testing dataset size: ', self.test_dataset.shape)
        print('Number of Entities: ', self.num_entity)
        print('Number of relations: ', self.num_relation)

    def get_dataset(self):
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

class Draw:
    def __init__(self, fig_save_folder='', tsne_folder=''):
        self.metric_ret_map = {'Training Loss':0,
                               'MR':1,
                               'MRR':2,
                               'Hits@10':3,
                               'num_epoch':4,
                               'init_epoch':5
                               }
        self.fig_save_folder = fig_save_folder + '/'
        self.tsne_folder = tsne_folder + '/'
        self.check_folder(folder=fig_save_folder)
        self.check_folder(folder=tsne_folder)
        self.check_file(file_path=self.tsne_folder,
                        file_name='tsne_meta_data.json')
        matplotlib.rcParams.update({'font.size': 22})

    def check_folder(self, folder):
        exist_dir = [dir_name for dir_name in os.listdir('./') if os.path.isdir('./' + dir_name)]
        if folder not in exist_dir:
            os.mkdir(folder)

    def check_file(self, file_path, file_name):
        file_list = [f_name for f_name in os.listdir('./' + file_path)]
        if file_name not in file_list:
            with open(file_path + file_name, 'w') as json_file:
                json.dump({}, json_file, indent=4)
                json_file.close()

    def meta_data_to_list(self, exp_dir_name):
        with open(exp_dir_name + 'meta_data.json', 'r') as json_file:
            meta_data = json.load(json_file)
            start_model = meta_data['global']['start model']
            if '.pt' in start_model:
                init_epoch = int(re.findall('\d+', start_model)[1])
            else:
                init_epoch = 0
            all_metrics = [metrics for metrics in meta_data['local'].values()]
            num_epoch = len(all_metrics)
            train_loss = [float(tr_loss['Average Training Loss']) for tr_loss in all_metrics]
            mr_val = [float(mr['MR']) for mr in all_metrics]
            mrr_val = [float(mrr['MRR']) for mrr in all_metrics]
            hits_at_10_val = [float(hits['Hits@10']) for hits in all_metrics]
            json_file.close()
        return train_loss, mr_val, mrr_val, hits_at_10_val, num_epoch, init_epoch

    def plot_metrics(self, met_dict, title, ylabel, max_epoch, en_save):
        plt.close()
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        for exp in met_dict.keys():
            all_metrics = self.meta_data_to_list(exp_dir_name=met_dict[exp])
            num_epoch = all_metrics[self.metric_ret_map['num_epoch']]
            metric = all_metrics[self.metric_ret_map[ylabel]]
            init_epoch = all_metrics[self.metric_ret_map['init_epoch']]
            end_epoch = init_epoch + num_epoch
            if end_epoch > max_epoch:
                end_epoch = max_epoch
            x_axis_epochs = list(range(init_epoch + 1, end_epoch+1))
            metric = metric[:len(x_axis_epochs)]
            plt.plot(x_axis_epochs, metric, label=exp)
        plt.legend(loc='center right')
        if en_save:
            fig = plt.gcf()
            fig.set_size_inches(15, 10)
            plt.savefig(self.fig_save_folder + title + ' ' + ylabel, dpi=200)
        else:
            plt.show()

    def plot_mr(self, mr_dict, title, max_epoch, en_save=False):
        self.plot_metrics(met_dict=mr_dict,
                          title=title,
                          ylabel='MR',
                          max_epoch=max_epoch,
                          en_save=en_save)

    def plot_mrr(self, mrr_dict, title, max_epoch, en_save=False):
        self.plot_metrics(met_dict=mrr_dict,
                          title=title,
                          ylabel='MRR',
                          max_epoch=max_epoch,
                          en_save=en_save)

    def plot_hits(self, hits_dict, title, max_epoch, en_save=False):
        self.plot_metrics(met_dict=hits_dict,
                          title=title,
                          ylabel='Hits@10',
                          max_epoch=max_epoch,
                          en_save=en_save)

    def plot_tr_loss(self, tr_dict, title, max_epoch, en_save=False):
        self.plot_metrics(met_dict=tr_dict,
                          title=title,
                          ylabel='Training Loss',
                          max_epoch=max_epoch,
                          en_save=en_save)

    def cal_tsne(self, model_path, title):
        print('Starting TSNE calculation: ')
        print('Model: ', model_path)
        transe_model = torch.load(model_path)
        entity_emb, relation_emb = transe_model['cur_model'].get_embeddings()
        tsne_emb = TSNE(n_components=2, perplexity=30, learning_rate='auto', n_iter=1000, init='random').fit_transform(entity_emb.cpu())
        print('Done!!')
        with open(self.tsne_folder + title + '.npy', 'wb') as numpy_file:
            np.save(numpy_file, tsne_emb)
            numpy_file.close()
        print('Saved TSNE data to disk.')

    def err_entity_threshold(self, err_entity_tensor, err_entity_datatype, threshold, title, dis_option):
        unique_entities = err_entity_tensor.unique()
        entity_err_freq_dict = {}
        for entity in unique_entities:
            freq_val = torch.count_nonzero(torch.where(err_entity_tensor == entity, 1, 0))
            entity_err_freq_dict[entity] = freq_val
        entity_err_freq_dict = dict(sorted(entity_err_freq_dict.items(), key=lambda item: item[1], reverse=True))
        total_errs = sum(entity_err_freq_dict.values())
        total_num_unique_entities = len(entity_err_freq_dict)
        entity_err_freq_dict = {k: v for k, v in entity_err_freq_dict.items() if v > threshold}
        select_errs = sum(entity_err_freq_dict.values())
        select_num_unique_entities = len(entity_err_freq_dict)
        print('percentage of errors accounted for: ', select_errs / total_errs)
        print('percentage of entities accounted for: ', select_num_unique_entities / total_num_unique_entities)
        with open(self.tsne_folder + 'tsne_meta_data.json', 'r+') as json_file:
            tsne_meta_data = json.load(json_file)
            tsne_meta_data[title][err_entity_datatype][dis_option]['threshold'] = int(threshold)
            tsne_meta_data[title][err_entity_datatype][dis_option]['ratio of errors over threshold'] = float(select_errs / total_errs)
            tsne_meta_data[title][err_entity_datatype][dis_option]['ratio of entities over threshold'] = float(select_num_unique_entities / total_num_unique_entities)
            json_file.seek(0)
            json.dump(tsne_meta_data, json_file, indent=4)
            json_file.truncate()
            json_file.close()
        return entity_err_freq_dict

    def plot_tsne(self, title, err_entity, err_entity_datatype, threshold, en_save=False):
        print('Choose an option: ')
        err_entity_keys_list = list(err_entity.keys())
        for index, key in enumerate(err_entity_keys_list):
            print(str(index+1) + ') ' + str(key))
        select_dis_option = int(input('Enter an option: '))
        if select_dis_option <= 0 or select_dis_option > len(err_entity_keys_list):
            return -1
        dis_option = err_entity_keys_list[select_dis_option - 1]
        tsne_emb = np.load(self.tsne_folder + title + '.npy')
        plt.close()
        fig = plt.gcf()
        fig.set_size_inches(30, 25)
        plt.title(title)
        plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1])
        entity_err_freq_dict = self.err_entity_threshold(err_entity_tensor=err_entity[dis_option],
                                                         err_entity_datatype=err_entity_datatype,
                                                         threshold=threshold,
                                                         title=title,
                                                         dis_option=dis_option)
        plt.scatter(tsne_emb[list(entity_err_freq_dict.keys()), 0], tsne_emb[list(entity_err_freq_dict.keys()), 1], color='red')
        if en_save:
            plt.savefig(self.fig_save_folder + title + '_' + dis_option, dpi=400)
        else:
            plt.show()