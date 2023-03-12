import copy
import time

import torch
from torch.utils.data import TensorDataset, DataLoader
from train_and_evaluate import SaveData


class TransEBoost(SaveData):
    def __init__(self, start_epoch, end_epoch, train_data, seed, device, batch_size, model, optimizer, num_entity,
                 folder, save_epoch=1):
        super().__init__(folder)
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.train_data = train_data
        self.seed = seed
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.optimizer = optimizer
        self.num_entity = num_entity
        # self.folder = folder
        self.save_epoch = save_epoch
        self.tail_dict = {}
        self.head_dict = {}
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.test_triplet = torch.zeros(self.num_entity, 3, dtype=torch.int64).to(self.device, non_blocking=True)
        self.all_entities = torch.arange(0, self.num_entity, dtype=torch.int64).to(self.device, non_blocking=True)

    def tensor_to_dataloader(self, data):
        # return a dataloader with the following settings
        return DataLoader(TensorDataset(data),
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True)

    def get_ranking_list(self, all_head, triplet):
        self.test_triplet[:, 1] = triplet[1]
        compare_val = None
        # select head or tail option
        if all_head:
            compare_val = triplet[0]
            self.test_triplet[:, 0] = self.all_entities
            self.test_triplet[:, 2] = triplet[2]
        else:
            compare_val = triplet[2]
            self.test_triplet[:, 0] = triplet[0]
            self.test_triplet[:, 2] = self.all_entities
        # get the prediction
        ranked_entities = self.model.predict(self.test_triplet)
        # get the rank of the correct entity from prediction
        pos = torch.where(ranked_entities == compare_val)[0]
        # get entities ranked higher than the correct ranks
        entities_ranked_higher = ranked_entities[:pos]
        del ranked_entities
        # add to rank to adjust
        rank = pos + 1
        return rank, entities_ranked_higher

    def evaluate(self, data):
        self.tail_dict = {}
        self.head_dict = {}
        for index, triplet in enumerate(data):
            # for tail:
            tail_rank, tail_entities_ranked_higher = self.get_ranking_list(all_head=False, triplet=triplet)
            # if ranked higher than 10, then note down the entities ranked higher than the correct entity
            if tail_rank > 10:
                self.tail_dict[index] = torch.cat(
                    (self.tail_dict.get(index, torch.tensor([], dtype=torch.int64, device=self.device)),
                     tail_entities_ranked_higher)).unique()
            # for head:
            head_rank, head_entities_ranked_higher = self.get_ranking_list(all_head=True, triplet=triplet)
            # if ranked higher than 10, then note down the entities ranked higher than the correct entity
            if head_rank > 10:
                self.head_dict[index] = torch.cat(
                    (self.head_dict.get(index, torch.tensor([], dtype=torch.int64, device=self.device)),
                     head_entities_ranked_higher)).unique()

    def sample_corr_batch(self, sample_batch):
        sample_batch = sample_batch.clone().detach()
        # choose head or tail to replace
        head_or_tail = torch.randint(0, 2, (1,))
        if head_or_tail == 0:  # replace tail
            tail_dict_keys = self.tail_dict.keys()
            for index, triplet in enumerate(sample_batch):
                # if an errored triplet, then sample from entities ranked higher than it 70 % of the time,
                # else sample randomly some entity.
                if index in tail_dict_keys:
                    temp = torch.randint(1, 11, (1,))
                    if temp <= 7:
                        dict_len = self.tail_dict[index].shape[0]
                        sample_batch[index, 2] = self.tail_dict[index][torch.randint(0, dict_len, (1,))]
                    else:
                        sample_batch[index, 2] = torch.randint(0, self.num_entity, (1,))
                else:
                    sample_batch[index, 2] = torch.randint(0, self.num_entity, (1,))
        else:  # replace head
            head_dict_keys = self.head_dict.keys()
            for index, triplet in enumerate(sample_batch):
                # if an errored triplet, then sample from entities ranked higher than it 70 % of the time,
                # else sample randomly some entity.
                if index in head_dict_keys:
                    temp = torch.randint(1, 11, (1,))
                    if temp <= 7:
                        dict_len = self.head_dict[index].shape[0]
                        sample_batch[index, 0] = self.head_dict[index][torch.randint(0, dict_len, (1,))]
                    else:
                        sample_batch[index, 0] = torch.randint(0, self.num_entity, (1,))
                else:
                    sample_batch[index, 0] = torch.randint(0, self.num_entity, (1,))
        return sample_batch

    def train(self):
        print('Starting TransE boosting Training: ')
        # loop for a number of epochs
        for epoch in range(self.start_epoch, self.end_epoch + 1):
            print('Starting epoch: ', epoch)
            avg_train_loss = 0
            start_t = time.time()
            # loop through the training dataset in batches
            for sample_batch in self.tensor_to_dataloader(self.train_data):
                # send sample batch to the gpu
                sample_batch = sample_batch[0].to(self.device, non_blocking=True)
                # evaluate to find the errored index and triplet ranked higher than the correct entity
                self.evaluate(data=sample_batch)
                # do corrupt triplet sampling based on the error information from evaluate
                corr_sample_batch = self.sample_corr_batch(sample_batch=sample_batch).to(self.device, non_blocking=True)
                # predict using the model and get loss
                loss = self.model(sample_batch, corr_sample_batch)
                avg_train_loss += loss.sum()
                # optimize embeddings using SGD
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            print(epoch, 'boost epoch is done')
            end_t = time.time()
            time_taken = float(end_t - start_t) / 60.0
            avg_train_loss = avg_train_loss / self.train_data.shape[0]
            print('Average Training loss is: ', avg_train_loss)
            print('Time taken for this epoch is (in mins.) : ', time_taken)
            if epoch % self.save_epoch == 0:
                self.save(model={'cur_model': self.model},
                          epoch=epoch,
                          avg_loss=avg_train_loss)


class TransEBoost2(SaveData):
    def __init__(self, pre_model, cur_model, train_data, start_epoch, end_epoch, total_epoch, seed, device, batch_size,
                 start_model, end_model, optimizer, num_entity, folder, save_epoch=1):
        self.pre_model = pre_model
        self.cur_model = cur_model
        self.train_data = train_data
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.seed = seed
        self.device = device
        self.batch_size = batch_size
        self.start_model = start_model
        self.end_model = end_model
        self.optimizer = optimizer
        self.num_entity = num_entity
        self.folder = folder
        self.save_epoch = save_epoch
        self.total_epoch = total_epoch
        self.tail_dict = {}
        self.head_dict = {}
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.test_triplet = torch.zeros(self.num_entity, 3, dtype=torch.int64).to(self.device, non_blocking=True)
        self.all_entities = torch.arange(0, self.num_entity, dtype=torch.int64).to(self.device, non_blocking=True)
        self.err_index = self.get_err_index(data=self.train_data, model=self.pre_model)

    def tensor_to_dataloader(self, data):
        # return the dataloader. it does not shuffle the training dataset.
        return DataLoader(TensorDataset(data),
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True)

    def get_ranking_list(self, all_head, triplet, model, get_err_entities=True):
        self.test_triplet[:, 1] = triplet[1]
        # select either head or tail to rank
        if all_head:
            compare_val = triplet[0]
            self.test_triplet[:, 0] = self.all_entities
            self.test_triplet[:, 2] = triplet[2]
        else:
            compare_val = triplet[2]
            self.test_triplet[:, 0] = triplet[0]
            self.test_triplet[:, 2] = self.all_entities
        # get the ranks using the model
        ranked_entities = model.predict(self.test_triplet)
        # get the rank of the correct entity
        pos = torch.where(ranked_entities == compare_val)[0]
        # if get falsely ranked entity is true then compute them
        if get_err_entities:
            entities_ranked_higher = ranked_entities[:pos]
        del ranked_entities
        # adjust to get proper rank
        rank = pos + 1
        # return based on condition to get falsely higher ranked entities or not.
        if get_err_entities:
            return rank, entities_ranked_higher
        else:
            return rank

    def evaluate(self, data, model, print_cond=False):
        self.tail_dict = {}
        self.head_dict = {}
        for index, triplet in enumerate(data):
            # for tail:
            tail_rank, tail_entities_ranked_higher = self.get_ranking_list(all_head=False, model=model, triplet=triplet)
            # if ranked higher than 10, then note down the entities ranked higher than the correct entity
            if tail_rank > 10:
                self.tail_dict[index] = torch.cat(
                    (self.tail_dict.get(index, torch.tensor([], dtype=torch.int64, device=self.device)),
                     tail_entities_ranked_higher)).unique()

            # for head:
            head_rank, head_entities_ranked_higher = self.get_ranking_list(all_head=True, model=model, triplet=triplet)
            # if ranked higher than 10, then note down the entities ranked higher than the correct entity
            if head_rank > 10:
                self.head_dict[index] = torch.cat(
                    (self.head_dict.get(index, torch.tensor([], dtype=torch.int64, device=self.device)),
                     head_entities_ranked_higher)).unique()

    def get_err_index(self, data, model):
        err_index = []
        for index, triplet in enumerate(data):
            # for tail rank:
            tail_rank = self.get_ranking_list(all_head=False, model=model, triplet=triplet, get_err_entities=False)

            # for head rank:
            head_rank = self.get_ranking_list(all_head=True, model=model, triplet=triplet, get_err_entities=False)
            # if the rank of the correct entity for head or tail prediciton is greater than 10 than
            # then get its index.
            if head_rank > 10 or tail_rank > 10:
                err_index.append(index)

            if index % 50000 == 0:
                print('Evaluated error index till: ', index)
        return torch.tensor(err_index)

    def sample_corr_batch(self, sample_batch):
        sample_batch = sample_batch.clone().detach()
        head_or_tail = torch.randint(0, 2, (1,))
        if head_or_tail == 0:  # replace tail
            tail_dict_keys = self.tail_dict.keys()
            for index, triplet in enumerate(sample_batch):
                # for the errored index sample from entities falsely ranked
                if index in tail_dict_keys:
                    dict_len = self.tail_dict[index].shape[0]
                    sample_batch[index, 2] = self.tail_dict[index][torch.randint(0, dict_len, (1,))]
                else:
                    sample_batch[index, 2] = torch.randint(0, self.num_entity, (1,))
        else:  # replace head
            head_dict_keys = self.head_dict.keys()
            for index, triplet in enumerate(sample_batch):
                # for the errored index sample from entities falsely ranked
                if index in head_dict_keys:
                    dict_len = self.head_dict[index].shape[0]
                    sample_batch[index, 0] = self.head_dict[index][torch.randint(0, dict_len, (1,))]
                else:
                    sample_batch[index, 0] = torch.randint(0, self.num_entity, (1,))
        return sample_batch

    def train(self):
        print('Starting TransEBoost2 training:')
        err_index = self.err_index
        # outer training loop. also previous model is set in this loop
        for model_num in range(self.start_model, self.end_model + 1):
            # inner training loop, that goes over epochs for current model
            for epoch in range(self.start_epoch, self.end_epoch + 1):
                print('Starting epoch: ', epoch)
                start_t = time.time()
                avg_train_loss = 0
                for sample_batch in self.tensor_to_dataloader(self.train_data[err_index, :]):
                    sample_batch = sample_batch[0].to(self.device, non_blocking=True)
                    # evaluate to get errored indexes
                    self.evaluate(data=sample_batch, model=self.pre_model)
                    # sample the corrupt triplets using the in formation from evaluation step
                    corr_sample_batch = self.sample_corr_batch(sample_batch=sample_batch).to(self.device,
                                                                                             non_blocking=True)
                    # use the model to get score to the triplet
                    loss = self.cur_model(sample_batch, corr_sample_batch)
                    avg_train_loss += loss.sum()
                    # use SGD to optimize the embeddings.
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    self.optimizer.step()
                print(epoch, 'boost epoch is done')
                end_t = time.time()
                avg_train_loss = avg_train_loss / self.train_data.shape[0]
                time_taken = float(end_t - start_t) / 60.0
                print('Average Training loss is: ', avg_train_loss)
                print('Time taken for this epoch is (in mins.) : ', time_taken)
                self.total_epoch += 1
                if self.total_epoch % self.save_epoch == 0:
                    model_dict = {'pre_model': self.pre_model,
                                  'cur_model': self.cur_model
                                  }
                    self.save(model=model_dict,
                              epoch=self.total_epoch,
                              avg_loss=avg_train_loss)
                              #cur_model_num=model_num)
            # set the previous model
            self.pre_model = copy.deepcopy(self.cur_model)
            pre_err_index = err_index
            # calculate the new error indexes
            err_index = self.get_err_index(self.train_data[err_index, :], model=self.cur_model)
            err_index = pre_err_index[err_index]
