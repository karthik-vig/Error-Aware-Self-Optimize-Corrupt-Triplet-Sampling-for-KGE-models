import torch
from pykeen.datasets import FB15k
from models import TransE
from train_and_evaluate import TransE_train_evaluate

#set transe model paratmeters:
device='cuda'
num_entity=14951
num_relation=1345
emb_dim=50
gamma=1

#set transe optimizer parameters:
lr=0.01
weight_decay=0

#transe training parameters:
batch_size=36
epoch=100

#set torch seeds:
seed=2022
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    #get training, validation and testing tensors from fb15k
    fb15k_dataset = FB15k()
    fb15k_train_dataset = fb15k_dataset.training.mapped_triples
    fb15k_val_dataset = fb15k_dataset.validation.mapped_triples
    fb15k_test_dataset = fb15k_dataset.testing.mapped_triples

    #print their shapes
    print('FB15K training dataset size: ', fb15k_train_dataset.shape)
    print('FB15K validation dataset size: ', fb15k_val_dataset.shape)
    print('FB15K testing dataset size: ', fb15k_test_dataset.shape)

    select_train_model = input('Train a TransE model?(y/n) : ')

    if select_train_model == 'y':

        continue_train = input('Continue training a saved model? (y/n) : ')

        if continue_train == 'y':

            print('Loading a TransE model from disk...')
            transe_model = torch.load('transe_model.pt')
            print('Done!')

        elif continue_train == 'n':

            #Create the transe model:
            transe_model = TransE(device=device,
                                 num_entity=num_entity,
                                 num_relation=num_relation,
                                 emb_dim=emb_dim,
                                 gamma=gamma,
                                 seed=seed)

        #Create the optimizer:
        optimizer = torch.optim.SGD(transe_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

        #Train the model:
        transe_model_train_eva = TransE_train_evaluate(train_dataset=fb15k_train_dataset,
                                                      val_dataset=fb15k_val_dataset,
                                                      batch_size=batch_size,
                                                      num_entity=num_entity,
                                                      model=transe_model,
                                                      device=device,
                                                      optimizer=optimizer,
                                                      epoch=epoch,
                                                      seed=seed)
        #train TransE model:
        transe_model_train_eva.train_transe()

        #Evaluate TransE model:
        transe_model_train_eva.evaluate_model()

        select_save_model = input('Save model ?(y/n) :')
        if select_save_model == 'y':
            torch.save(transe_model, 'transe_model.pt')
            print('Saving done!')


    elif select_train_model == 'n':

        print('Loading a TransE model from disk...')
        transe_model = torch.load('transe_model.pt')
        print('Done!')

        transe_model_train_eva = TransE_train_evaluate(train_dataset=fb15k_train_dataset,
                                                      val_dataset=fb15k_val_dataset,
                                                      batch_size=None,
                                                      num_entity=num_entity,
                                                      model=transe_model,
                                                      device=device,
                                                      optimizer=None,
                                                      epoch=None,
                                                      seed=seed)

        # Evaluate TransE model:
        transe_model_train_eva.evaluate_model()