import torch
from torch.utils.data import TensorDataset, DataLoader

class TransEBoost():
    def __init__(self, start_epoch, end_epoch, train_data, val_data, seed, device, batch_size, model, optimizer, num_entity):
        self.start_epoch=start_epoch
        self.end_epoch=end_epoch
        self.train_data=train_data
        self.val_data=val_data
        self.seed=seed
        self.device=device
        self.batch_size=batch_size
        self.model=model
        self.optimizer=optimizer
        self.num_entity=num_entity

        self.tail_dict={}
        self.head_dict={}

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.test_triplet=torch.zeros(self.num_entity, 3, dtype=torch.int64).to(self.device, non_blocking=True)
        self.all_entities=torch.arange(0, self.num_entity, dtype=torch.int64).to(self.device, non_blocking=True)
        self.score_tensor=torch.arange(self.batch_size * 2, dtype=torch.float64).to(self.device, non_blocking=True)

    def tensor_to_dataloader(self, data):
        return DataLoader(TensorDataset(data),
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True)

    def get_ranking_list(self, all_head, triplet):
        self.test_triplet[:, 1]=triplet[1]
        compare_val = None
        if all_head:
            compare_val=triplet[0]
            self.test_triplet[:, 0]=self.all_entities
            self.test_triplet[:, 2]=triplet[2]
        else:
            compare_val=triplet[2]
            self.test_triplet[:, 0]=triplet[0]
            self.test_triplet[:, 2]=self.all_entities

        ranked_entities = self.model.predict(self.test_triplet)
        pos = torch.where(ranked_entities == compare_val)[0]
        entities_ranked_higher=ranked_entities[:pos]
        del ranked_entities
        rank=pos + 1

        return rank, entities_ranked_higher

    def evaluate(self, data, print_cond=False):
        #print('Starting evaluation for TransE boosting: ')
        rank_count=0
        self.tail_dict = {}
        self.head_dict = {}
        for index, triplet in enumerate(data):
            #for tail:
            tail_rank, tail_entities_ranked_higher = self.get_ranking_list(all_head=False, triplet=triplet)
            #tail_rank+=1 #error
            self.score_tensor[rank_count]=tail_rank
            if tail_rank > 10:
                self.tail_dict[index] = torch.cat(( self.tail_dict.get(index, torch.tensor([], dtype=torch.int64,device=self.device) ),
                                                    tail_entities_ranked_higher)).unique()

            #for head:
            head_rank, head_entities_ranked_higher = self.get_ranking_list(all_head=True, triplet=triplet)
            #head_rank+=1 #error
            self.score_tensor[rank_count + 1]=head_rank
            if head_rank > 10:
                self.head_dict[index] = torch.cat(( self.head_dict.get(index, torch.tensor([], dtype=torch.int64, device=self.device)),
                                                    head_entities_ranked_higher)).unique()

            rank_count+=2

        mr=torch.mean(self.score_tensor)
        mrr=torch.reciprocal(self.score_tensor).mean()
        hits_at_10=torch.where(self.score_tensor < 11.0, 1.0, 0.0).mean()

        if print_cond:
            print('MR: ', mr)
            print('MRR: ', mrr)
            print('Hits@10: ', hits_at_10)


    def sample_corr_batch(self, sample_batch):
        sample_batch = sample_batch.clone().detach()
        # offset=0
        # if self.batch_size % 2 != 0:
        #     offset=1
        head_or_tail=torch.randint(0, 2, (1,))

        if head_or_tail == 0: #replace tail
            tail_dict_keys=self.tail_dict.keys()
            for index, triplet in enumerate(sample_batch):
                if index in tail_dict_keys:
                    temp = torch.randint(1, 11, (1,))
                    if temp <= 7:
                        dict_len=self.tail_dict[index].shape[0]
                        sample_batch[index, 2]=self.tail_dict[index][torch.randint(0, dict_len, (1,))]
                    else:
                        sample_batch[index, 2]=torch.randint(0, self.num_entity, (1,))
                else:
                    sample_batch[index, 2]=torch.randint(0, self.num_entity, (1,))

        else: #replace head
            head_dict_keys = self.head_dict.keys()
            for index, triplet in enumerate(sample_batch):
                if index in head_dict_keys:
                    temp = torch.randint(1, 11, (1,))
                    if temp <= 7:
                        dict_len = self.head_dict[index].shape[0]
                        # print(dict_len) # remove later
                        sample_batch[index, 0] = self.head_dict[index][torch.randint(0, dict_len, (1,))]
                    else:
                        sample_batch[index, 0] = torch.randint(0, self.num_entity, (1,))
                else:
                    sample_batch[index, 0] = torch.randint(0, self.num_entity, (1,))

        return sample_batch

    def train(self):
        print('Starting TransE boosting Training: ')
        for epoch in range(self.start_epoch, self.end_epoch + 1):
            print('Starting epoch: ', epoch)
            avg_train_loss=0
            for sample_batch in self.tensor_to_dataloader(self.train_data):
                sample_batch = sample_batch[0].to(self.device, non_blocking=True)
                self.evaluate(data=sample_batch, print_cond=False)
                corr_sample_batch = self.sample_corr_batch(sample_batch=sample_batch).to(self.device, non_blocking=True)
                loss = self.model(sample_batch, corr_sample_batch)
                avg_train_loss+=loss.sum()
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

            print(epoch, 'boost epoch is done')
            print('Average Training loss is: ', avg_train_loss / self.train_data.shape[0])
            torch.save(self.model, 'transe_boosted_models/transe_boost_model_'+str(epoch)+'.pt')



class TransEBoost2():
    def __init__(self, pre_model, cur_model, train_data, start_epoch, end_epoch, seed, device, batch_size, start_model,
                 end_model, optimizer, num_entity):
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
        self.tail_dict = {}
        self.head_dict = {}
        self.err_index = None

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.test_triplet=torch.zeros(self.num_entity, 3, dtype=torch.int64).to(self.device, non_blocking=True)
        self.all_entities=torch.arange(0, self.num_entity, dtype=torch.int64).to(self.device, non_blocking=True)
        # self.score_tensor=torch.arange(self.batch_size * 2, dtype=torch.float64).to(self.device, non_blocking=True)

    def set_err_index(self, err_index):
        self.err_index = err_index

    def tensor_to_dataloader(self, data):
        return DataLoader(TensorDataset(data),
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True)

    def get_ranking_list(self, all_head, triplet, model, get_err_entities=True):
        self.test_triplet[:, 1]=triplet[1]
        if all_head:
            compare_val=triplet[0]
            self.test_triplet[:, 0]=self.all_entities
            self.test_triplet[:, 2]=triplet[2]
        else:
            compare_val=triplet[2]
            self.test_triplet[:, 0]=triplet[0]
            self.test_triplet[:, 2]=self.all_entities

        ranked_entities = model.predict(self.test_triplet)
        pos = torch.where(ranked_entities == compare_val)[0]
        if get_err_entities:
            entities_ranked_higher=ranked_entities[:pos]
        del ranked_entities
        rank=pos + 1
        if get_err_entities:
            return rank, entities_ranked_higher
        else:
            return rank

    def evaluate(self, data, model, print_cond=False):
        #print('Starting evaluation for TransE boosting: ')
        # rank_count=0
        self.tail_dict = {}
        self.head_dict = {}
        for index, triplet in enumerate(data):
            #for tail:
            tail_rank, tail_entities_ranked_higher = self.get_ranking_list(all_head=False, model=model, triplet=triplet)
            # self.score_tensor[rank_count]=tail_rank
            if tail_rank > 10:
                self.tail_dict[index] = torch.cat(( self.tail_dict.get(index, torch.tensor([], dtype=torch.int64,device=self.device) ),
                                                    tail_entities_ranked_higher)).unique()

            #for head:
            head_rank, head_entities_ranked_higher = self.get_ranking_list(all_head=True, model=model, triplet=triplet)
            # self.score_tensor[rank_count + 1]=head_rank
            if head_rank > 10:
                self.head_dict[index] = torch.cat(( self.head_dict.get(index, torch.tensor([], dtype=torch.int64, device=self.device)),
                                                    head_entities_ranked_higher)).unique()
            # rank_count+=2
        # mr=torch.mean(self.score_tensor)
        # mrr=torch.reciprocal(self.score_tensor).mean()
        # hits_at_10=torch.where(self.score_tensor < 11.0, 1.0, 0.0).mean()
        # if print_cond:
        #     print('MR: ', mr)
        #     print('MRR: ', mrr)
        #     print('Hits@10: ', hits_at_10)

    def get_err_index(self, data, model):
        err_index = []
        for index, triplet in enumerate(data):
            #for tail rank:
            tail_rank = self.get_ranking_list(all_head=False, model=model, triplet=triplet, get_err_entities=False)

            #for head rank:
            head_rank = self.get_ranking_list(all_head=True, model=model, triplet=triplet, get_err_entities=False)

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
                if index in tail_dict_keys:
                    dict_len = self.tail_dict[index].shape[0]
                    sample_batch[index, 2] = self.tail_dict[index][torch.randint(0, dict_len, (1,))]
                else:
                    sample_batch[index, 2] = torch.randint(0, self.num_entity, (1,))
        else:  # replace head
            head_dict_keys = self.head_dict.keys()
            for index, triplet in enumerate(sample_batch):
                if index in head_dict_keys:
                    dict_len = self.head_dict[index].shape[0]
                    sample_batch[index, 0] = self.head_dict[index][torch.randint(0, dict_len, (1,))]
                else:
                    sample_batch[index, 0] = torch.randint(0, self.num_entity, (1,))
        return sample_batch

    def train(self):
        print('Starting TransEBoost2 training:')
        # err_index = torch.arange(0, self.train_data.shape[0], dtype=torch.int64).to(self.device,
        #                                                                             non_blocking=True)
        if self.err_index == None:
            print('err_index not set. Aborting.')
            return
        else:
            err_index = self.err_index

        for model in range(self.start_model, self.end_model):
            for epoch in range(self.start_epoch, self.end_epoch + 1):
                print('Starting epoch: ', epoch)
                avg_train_loss = 0
                for sample_batch in self.tensor_to_dataloader(self.train_data[err_index,:]):
                    sample_batch = sample_batch[0].to(self.device, non_blocking=True)
                    self.evaluate(data=sample_batch, model=self.pre_model)
                    corr_sample_batch = self.sample_corr_batch(sample_batch=sample_batch).to(self.device,
                                                                                             non_blocking=True)
                    loss = self.cur_model(sample_batch, corr_sample_batch)
                    avg_train_loss += loss.sum()
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    self.optimizer.step()
                print(epoch, 'boost epoch is done')
                print('Average Training loss is: ', avg_train_loss / self.train_data.shape[0])
            torch.save(self.cur_model, 'transe_boosted2_models/transe_boost2_model_'+str(model)+'.pt')
            self.pre_model = self.cur_model
            pre_err_index = err_index
            err_index = self.get_err_index(self.train_data[err_index,:], model=self.cur_model)
            err_index = pre_err_index[err_index]



