import torch
from torch.utils.data import TensorDataset, DataLoader

def TransEBoost():
    def __init__(self, epoch, train_data, val_data, seed, device, batch_size, model, num_entity):
        self.epoch=epoch
        self.train_data=train_data
        self.val_data=val_data
        self.seed=seed
        self.device=device
        self.batch_size=batch_size
        self.model=model
        self.num_entity=num_entity
        self.tail_dict={}
        self.head_dict={}

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.test_triplet=torch.zeros(self.num_entity, 3, dtype=torch.int64, non_blocking=True).to(self.device, non_blocking=True)
        self.all_entities=torch.arange(0, self.num_entity, dtype=torch.int64).to(self.device, non_blocking=True)
        self.score_tensor=torch.arange(self.batch_size, dtype=torch.float64).to(self.device, non_blocking=True)

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

    def evaluate(self, data):
        print('Starting TransE boosting: ')
        rank_count=0
        for index, triplet in enumerate(data):
            #for tail:
            tail_rank, tail_entities_ranked_higher = self.get_ranking_list(all_head=False, triplet=triplet)
            self.score_tensor[rank_count]=tail_rank
            self.tail_dict[index] = self.tail_dict.get(index, set()).union( set(tail_entities_ranked_higher.tolist()) )

            #for head:
            head_rank, head_entities_ranked_higher = self.get_ranking_list(all_head=True, triplet=triplet)
            self.score_tensor[rank_count + 1]=head_rank
            self.head_dict[index] = self.head_dict.get(index, set()).union( set(head_entities_ranked_higher.tolist()) )

            rank_count+=2

        mr=torch.mean(self.score_tensor)
        mrr=torch.reciprocal(self.score_tensor).mean()
        hits_at_10=torch.where(self.score_tensor < 11.0, 1.0, 0.0).mean()

        print('MR: ', mr)
        print('MRR: ', mrr)
        print('Hits@10: ', hits_at_10)


    def sample_corr_batch(self, sample_batch):
        offset=0
        if self.batch_size % 2 != 0:
            offset=1
        head_or_tail=torch.randint(0, 2, (1,))

        if head_or_tail == 0: #replace tail
            tail_dict_keys=self.tail_dict.keys()
            for index, triplet in enumerate(sample_batch):
                if index in tail_dict_keys:
                    temp = torch.randint(0, 10, (1,))
                    if temp <= 7:
                        dict_len=len(self.tail_dict[index])
                        sample_batch[index, 2]=list(self.tail_dict[index])[ torch.randint(0, dict_len, (1,)) ]
                    else:
                        sample_batch[index, 2]=torch.randint(0, self.num_entity, (1,))
                else:
                    sample_batch[index, 2]=torch.randint(0, self.num_entity, (1,))

        else: #replace head
            head_dict_keys = self.head_dict.keys()
            for index, triplet in enumerate(sample_batch):
                if index in head_dict_keys:
                    temp = torch.randint(0, 10, (1,))
                    if temp <= 7:
                        dict_len = len(self.head_dict[index])
                        sample_batch[index, 2] = list(self.head_dict[index])[torch.randint(0, dict_len, (1,))]
                    else:
                        sample_batch[index, 2] = torch.randint(0, self.num_entity, (1,))
                else:
                    sample_batch[index, 2] = torch.randint(0, self.num_entity, (1,))

    def train(self):
        print('Starting TransE boosting Training: ')
        avg_train_loss=0
        for i in range(1, self.epoch + 1):
            for sample_batch in self.tensor_to_dataloader(self.train_data):
                sample_batch = sample_batch[0].to(self.device, non_blocking=True)
                self.evaluate(sample_batch=sample_batch)
                corr_sample_batch = self.sample_corr_batch(sample_batch=sample_batch).to(self.device, non_blocking=True)
                loss = self.model(sample_batch, corr_sample_batch)
                avg_train_loss+=loss.sum()
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

            print(i, 'epoch is done')
            print('Average Training loss is: ', avg_train_loss / self.train_data.shape[0])
