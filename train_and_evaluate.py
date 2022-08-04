import torch
from torch.utils.data import TensorDataset, DataLoader


class TransETrain():
    def __init__(self, train_dataset, batch_size, num_entity, model, device, optimizer, epoch, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.num_entity = num_entity
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.epoch = epoch
        self.train_dataset_len = self.train_dataset.shape[0]
        self.train_data_loader = self.create_dataloader(self.train_dataset)

    def create_dataloader(self, dataset):
        tensor_dataset = TensorDataset(dataset)
        return DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def create_corr_triplet(self, sample_data):
        corr_triplet = sample_data.clone().detach()
        head_or_tail = torch.randint(0, 2, (1,))
        entity_tensor = torch.randint(0, self.num_entity, (sample_data.shape[0],))
        if head_or_tail == 0:
            corr_triplet[:, 0] = entity_tensor
        else:
            corr_triplet[:, 2] = entity_tensor
        return corr_triplet

    def train_transe(self):
        print('Starting Training:')
        for i in range(1, self.epoch + 1):
            avg_train_loss = 0
            for index, batch_data in enumerate(self.train_data_loader):
                sample_data = batch_data[0]
                sample_data = sample_data.to(self.device, non_blocking=True)
                corr_sample_data = self.create_corr_triplet(sample_data=sample_data).to(self.device, non_blocking=True)
                loss = self.model(sample_data, corr_sample_data)
                avg_train_loss += loss.sum()
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            print(i, 'epoch is done')
            print('Average Training loss is: ', avg_train_loss / self.train_dataset_len)
            # evaluate_model()



class Evaluation():
    def __init__(self, data, model, num_entity, device):
        self.dataset = data
        self.model = model
        self.num_entity = num_entity
        self.device = device
        self.all_entities = self.all_entities = torch.arange(0, self.num_entity, dtype=torch.int64).to(self.device, non_blocking=True)
        self.test_triplet = torch.zeros(self.num_entity, 3, dtype=torch.int64).to(self.device, non_blocking=True)

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
        del ranked_entities
        rank = pos + 1
        return rank

    def evaluate_model(self):
        print('Starting Evaluation:')
        tail_score_tensor = torch.zeros(self.dataset.shape[0], dtype=torch.float64).to(self.device, non_blocking=True)
        head_score_tensor = torch.zeros(self.dataset.shape[0], dtype=torch.float64).to(self.device, non_blocking=True)
        for index, triplet in enumerate(self.dataset):
            # calculate the tail rank:
            tail_rank = self.get_ranking_list(all_head=False, triplet=triplet.to(device=self.device, non_blocking=True))
            tail_score_tensor[index] = tail_rank
            # calculate the head rank:
            head_rank = self.get_ranking_list(all_head=True, triplet=triplet.to(device=self.device, non_blocking=True))
            head_score_tensor[index] = head_rank
            # to show progress
            if index % 10000 == 0:
                print('Triplets evaluated: ', index)
        # concat the two tesors, to produce a final listing of ranks
        score_tensor = torch.cat((tail_score_tensor, head_score_tensor))
        mr_score = torch.mean(score_tensor)
        mrr_score = torch.reciprocal(score_tensor).mean()
        hit_at_10_score = torch.where(score_tensor < 11.0, 1.0, 0.0).mean()
        print('Mean Rank for prediction is: ', mr_score)
        print('Mean Reciprocal Rank for prediction is: ', mrr_score)
        print('Hits@10 for prediction is: ', hit_at_10_score)
        return mr_score, mrr_score, hit_at_10_score

