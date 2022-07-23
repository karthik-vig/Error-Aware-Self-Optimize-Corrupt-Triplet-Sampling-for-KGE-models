import torch
from pykeen.datasets import FB15k237
from models import TransE
from train_and_evaluate import TransE_train_evaluate
# from boost import TransEBoost
from transe_boost import TransEBoost
from os import listdir

#set transe model paratmeters:
device='cuda'
num_entity=14505
num_relation=237
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

#save and load transe_models:
# evaluate_model='transe_model.pt'
evaluate_model='transe_boosted_models/transe_boost_model_1.pt'

def restore_model():
    file_list=listdir('transe_boosted_models/')
    file_numbers=[int(i.split('_')[-1].split('.')[-2]) for i in file_list]
    model_num=max(file_numbers)
    transe_model=torch.load('transe_boosted_models/transe_boost_model_'+str(model_num)+'.pt')
    start_epoch=model_num+1
    return transe_model, start_epoch

if __name__ == '__main__':
    #get training, validation and testing tensors from fb15k
    fb15k237_dataset = FB15k237()
    fb15k237_train_dataset = fb15k237_dataset.training.mapped_triples
    fb15k237_val_dataset = fb15k237_dataset.validation.mapped_triples
    fb15k237_test_dataset = fb15k237_dataset.testing.mapped_triples

    #print their shapes
    print('FB15K237 training dataset size: ', fb15k237_train_dataset.shape)
    print('FB15K237 validation dataset size: ', fb15k237_val_dataset.shape)
    print('FB15K237 testing dataset size: ', fb15k237_test_dataset.shape)

    select_train_model = input('''1) Train a TransE model? 
                               \n2) Evaluate a model? 
                               \n3) Resume Training a boosted model? 
                               \n Enter (1, 2, 3): ''')

    if select_train_model == '1':

        continue_train = input('Continue training a saved model? (y/n) : ')

        if continue_train == 'y':

            print('Loading a TransE model from disk...')
            transe_model = torch.load('transe_model.pt')
            print('Done!')

        elif continue_train == 'n':

            #Create the transe model:
            print('Creating a new TransE model: ')
            transe_model = TransE(device=device,
                                 num_entity=num_entity,
                                 num_relation=num_relation,
                                 emb_dim=emb_dim,
                                 gamma=gamma,
                                 seed=seed)
            print('Done!!')

        #Create the optimizer:
        optimizer = torch.optim.SGD(transe_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

        #Train the model:
        transe_model_train_eva = TransE_train_evaluate(train_dataset=fb15k237_train_dataset,
                                                       val_dataset=fb15k237_val_dataset,
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


    elif select_train_model == '2':

        print('Loading a TransE model from disk...')
        transe_model = torch.load(evaluate_model)
        print('Done!')

        print('Model being evaluated: ', evaluate_model)

        transe_model_train_eva = TransE_train_evaluate(train_dataset=fb15k237_train_dataset,
                                                       val_dataset=fb15k237_val_dataset,
                                                       batch_size=None,
                                                       num_entity=num_entity,
                                                       model=transe_model,
                                                       device=device,
                                                       optimizer=None,
                                                       epoch=None,
                                                       seed=seed)

        # Evaluate TransE model:
        transe_model_train_eva.evaluate_model()

    elif select_train_model == '3':

        #testing boost here:

        transe_model, start_epoch=restore_model()

        #Create the optimizer:
        optimizer = torch.optim.SGD(transe_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

        test_obj = TransEBoost(start_epoch=start_epoch,
                               end_epoch=10,
                               train_data=fb15k237_train_dataset,
                               val_data=fb15k237_val_dataset,
                               seed=2022,
                               device=device,
                               batch_size=36,
                               model=transe_model,
                               optimizer=optimizer,
                               num_entity=num_entity)

        test_obj.train()


