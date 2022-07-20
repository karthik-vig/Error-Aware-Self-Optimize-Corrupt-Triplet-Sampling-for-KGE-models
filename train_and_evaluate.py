import torch
from torch.utils.data import TensorDataset, DataLoader


class TransE_train_evaluate():
    def __init__(self, train_dataset, val_dataset, batch_size, num_entity, model, device, optimizer, epoch, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.train_dataset = train_dataset
        self.val_dataset = TensorDataset(val_dataset)
        self.batch_size = batch_size
        self.num_entity = num_entity
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.epoch = epoch

        self.val_dataset_len = len(self.val_dataset)
        self.train_dataset_len = self.train_dataset.shape[0]
        self.train_data_loader = self.create_dataloader(self.train_dataset)


    def create_dataloader(self, dataset):
        tensor_dataset = TensorDataset(dataset)
        return DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)


    def evaluate_model(self):
        print('Starting Evaluation:')
        score_tensor = torch.zeros(self.val_dataset_len * 2, dtype=torch.float64).to(self.device, non_blocking=True)
        epoch = 0
        rank_count = 0
        test_triplet = torch.zeros(self.num_entity, 3, dtype=torch.int64)
        entity_tensor = torch.arange(0, self.num_entity, dtype=torch.int64)
        for triplet in self.val_dataset:
            #calculate the tail rank:
            original_tail = triplet[0][2]
            test_triplet[:, 0] = triplet[0][0]
            test_triplet[:, 1] = triplet[0][1]
            test_triplet[:, 2] = entity_tensor
            test_triplet = test_triplet.to(self.device, non_blocking=True)
            sorted_score_index = self.model.predict(test_triplet)
            tail_rank = (sorted_score_index == original_tail).nonzero()[0, 0]
            del sorted_score_index
            tail_rank += torch.tensor(1)

            #calculate the head rank:
            original_head = triplet[0][0]
            test_triplet[:, 0] = entity_tensor
            test_triplet[:, 1] = triplet[0][1]
            test_triplet[:, 2] = triplet[0][2]
            test_triplet = test_triplet.to(self.device, non_blocking=True)
            sorted_score_index = self.model.predict(test_triplet)
            head_rank = (sorted_score_index == original_head).nonzero()[0, 0]
            del sorted_score_index
            head_rank += torch.tensor(1)

            # Add rank values to a tensor:
            score_tensor[rank_count] = tail_rank
            score_tensor[rank_count + 1] = head_rank
            rank_count += 2
            epoch += 1
            if epoch % 5000 == 0:
                print('Triplets evaluated: ', epoch)
        tail_mr_score = torch.mean(score_tensor)
        tail_mrr_score = torch.reciprocal(score_tensor).mean()
        tail_hit_at_10_score = torch.where(score_tensor < 11.0, 1.0, 0.0).mean()

        print('Mean Rank for prediction is: ', tail_mr_score)
        print('Mean Reciprocal Rank for prediction is: ', tail_mrr_score)
        print('Hits@10 for prediction is: ', tail_hit_at_10_score)



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