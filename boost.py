from train_and_evaluate import TransE_train_evaluate


class TransEBoost(TransE_train_evaluate):
    def __init__(self, train_dataset, val_dataset, batch_size, num_entity, model, device, optimizer, epoch, seed):
        super(TransEBoost, self).__init__(train_dataset=train_dataset,
                                          val_dataset=val_dataset,
                                          batch_size=batch_size,
                                          num_entity=num_entity,
                                          model=model,
                                          device=device,
                                          optimizer=optimizer,
                                          epoch=epoch,
                                          seed=seed)

        print(self.epoch)