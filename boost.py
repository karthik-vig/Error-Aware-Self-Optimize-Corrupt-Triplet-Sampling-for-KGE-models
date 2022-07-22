import random
import torch
from train_and_evaluate import TransE_train_evaluate


class TransEBoost(TransE_train_evaluate):
    def __init__(self, train_dataset, val_dataset, batch_size, num_entity, model, device, optimizer, epoch, seed, val_batch_size):
        super(TransEBoost, self).__init__(train_dataset=train_dataset,
                                          val_dataset=val_dataset,
                                          batch_size=batch_size,
                                          num_entity=num_entity,
                                          model=model,
                                          device=device,
                                          optimizer=optimizer,
                                          epoch=epoch,
                                          seed=seed)

        random.seed(seed)

        self.head_error_dict = {}
        self.tail_error_dict = {}
        self.head_error_entities = []
        self.tail_error_entities = []
        self.head_error_entity_len = 0
        self.tail_error_entity_len = 0
        self.error_index = []
        self.val_dataset_batch = self.val_dataset[:val_batch_size, :]
        self.val_dataset_batch_len = val_batch_size
        # self.head_error_index = []


    def boost_evaluate_model(self):
        print('Starting Evaluation for Boosting:')
        epoch = 0
        rank_count = 0
        test_triplet = torch.zeros(self.num_entity, 3, dtype=torch.int64)
        entity_tensor = torch.arange(0, self.num_entity, dtype=torch.int64)
        # tail_error_dict = {}
        # head_error_dict = {}
        score_tensor = torch.zeros(self.val_dataset_batch_len * 2, dtype=torch.float64).to(self.device, non_blocking=True)
        print(score_tensor.shape)
        for index, triplet in enumerate(self.val_dataset_batch):
            # calculate the tail rank:
            original_tail = triplet[0][2]
            test_triplet[:, 0] = triplet[0][0]
            test_triplet[:, 1] = triplet[0][1]
            test_triplet[:, 2] = entity_tensor
            test_triplet = test_triplet.to(self.device, non_blocking=True)
            sorted_score_index = self.model.predict(test_triplet)
            # tail_rank = (sorted_score_index == original_tail).nonzero()[0, 0]
            tail_rank = torch.where(sorted_score_index == original_tail)[0]
            #tail dict
            if tail_rank > 9:
                self.tail_error_dict[original_tail] = self.tail_error_dict.get(original_tail, set()).union(set(sorted_score_index[:tail_rank].tolist()))
                self.error_index.append(index)
            del sorted_score_index
            tail_rank += torch.tensor(1)

            # calculate the head rank:
            original_head = triplet[0][0]
            test_triplet[:, 0] = entity_tensor
            test_triplet[:, 1] = triplet[0][1]
            test_triplet[:, 2] = triplet[0][2]
            test_triplet = test_triplet.to(self.device, non_blocking=True)
            sorted_score_index = self.model.predict(test_triplet)
            # head_rank = (sorted_score_index == original_head).nonzero()[0, 0]
            head_rank = torch.where(sorted_score_index == original_head)[0]
            #dict
            if head_rank > 9:
                self.head_error_dict[original_head] = self.head_error_dict.get(original_head, set()).union(set(sorted_score_index[:head_rank].tolist()))
                self.error_index.append(index)
            del sorted_score_index
            head_rank += torch.tensor(1)

            # Add rank values to a tensor:
            print(rank_count)
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

        self.head_error_entities = self.head_error_dict.keys()
        self.tail_error_entities = self.head_error_dict.keys()
        self.head_error_entity_len = len(self.head_error_entities)
        self.tail_error_entity_len = len(self.tail_error_entities)

        #return self.error_index


    def create_corr_triplet(self, sample_data):
        corr_triplet = sample_data.clone().detach()
        head_or_tail = torch.randint(0, 2, (1,))
        entity_tensor = torch.randint(0, self.num_entity, (sample_data.shape[0],))
        if head_or_tail == 0:
            for entity in self.head_error_dict.keys():
                errored_entities_pos = torch.where(corr_triplet[:,0] == entity)[0].tolist()
                # errored_entities_len = len(errored_entities_pos)
                # list[self.head_error_dict[entity]] #[ torch.randint(0, self.head_error_entity_len, ( int( errored_entities_len / 2 ), )) ]
                if len(errored_entities_pos) % 2 != 0:
                    offset = 1
                else:
                    offset = 0
                corr_triplet[errored_entities_pos, 0] = random.choices(list[self.head_error_dict[entity]], k=int(len(errored_entities_pos) / 2) ) \
                                                        + random.choices(entity_tensor, k=int(len(errored_entities_pos) / 2) + offset)
                remain_entities_pos = set({[i for i in range(corr_triplet.shape[0])]}).difference(errored_entities_pos)
                corr_triplet[remain_entities_pos, 0] = torch.randint(0, self.num_entity, (len(remain_entities_pos),))
        else:
            for entity in self.tail_error_dict.keys():
                errored_entities_pos = torch.where(corr_triplet[:,2] == entity)[0].tolist()
                # errored_entities_len = len(errored_entities_pos)
                # list[self.head_error_dict[entity]] #[ torch.randint(0, self.head_error_entity_len, ( int( errored_entities_len / 2 ), )) ]
                if len(errored_entities_pos) % 2 != 0:
                    offset = 1
                else:
                    offset = 0
                corr_triplet[errored_entities_pos, 2] = random.choices(list[self.tail_error_dict[entity]], k=int(len(errored_entities_pos) / 2) ) \
                                                        + random.choices(entity_tensor, k=int(len(errored_entities_pos) / 2) + offset)
                remain_entities_pos = set({[i for i in range(corr_triplet.shape[0])]}).difference(errored_entities_pos)
                corr_triplet[remain_entities_pos, 2] = torch.randint(0, self.num_entity, (len(remain_entities_pos),))
        return corr_triplet


    def boost(self):
        temp_val = int( self.val_dataset_len / self.val_dataset_batch_len )
        if self.val_dataset_len % self.val_dataset_batch_len != 0:
            temp_val += 1
        for boost_epoch in range(1, 2):
            last_selection = 0
            count = 1
            for current_batch in range(1, temp_val+1):
                current_selection = self.val_dataset_batch_len*count
                self.val_dataset_batch = self.val_dataset[last_selection:current_selection, :]
                print(self.val_dataset[0].shape)
                self.val_dataset_batch_len = len(self.val_dataset_batch[0])
                print(self.val_dataset_batch_len)
                self.boost_evaluate_model()
                self.train_transe()
                last_selection = current_selection
                count += 1
            torch.save(self.model, 'trasne_boost_model'+str(boost_epoch)+'.pt')



