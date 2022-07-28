import re
import torch
from pykeen.datasets import FB15k237, FB15k, WN18
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

#TransEBoost parameters:
end_epoch=100

def main():

    select_dataset = input('''1)FB15K\n2)FB15K237\n3)WN18\nEnter 1,2 or 3:''')
    if select_dataset == '1':
        fb15k_dataset = FB15k()
        train_dataset = fb15k_dataset.training.mapped_triples
        val_dataset = fb15k_dataset.validation.mapped_triples
        test_dataset = fb15k_dataset.testing.mapped_triples
    elif select_dataset == '2':
        #get training, validation and testing tensors from fb15k
        fb15k237_dataset = FB15k237()
        train_dataset = fb15k237_dataset.training.mapped_triples
        val_dataset = fb15k237_dataset.validation.mapped_triples
        test_dataset = fb15k237_dataset.testing.mapped_triples
    elif select_dataset == '3':
        wn18_dataset = WN18()
        train_dataset = wn18_dataset.training.mapped_triples
        val_dataset = wn18_dataset.validation.mapped_triples
        test_dataset = wn18_dataset.testing.mapped_triples

    #print their shapes
    print('Training dataset size: ', train_dataset.shape)
    print('Validation dataset size: ', val_dataset.shape)
    print('Testing dataset size: ', test_dataset.shape)

    select_train_model = input('''1) Train a new TransE model?
                               \n2) Train a existing TransE model? 
                               \n3) Evaluate a model? 
                               \n4) Resume Training a boosted model? 
                               \n Enter (1, 2, 3, 4): ''')

    if select_train_model == '1':
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
        transe_model_train_eva = TransE_train_evaluate(train_dataset=train_dataset,
                                                       val_dataset=val_dataset,
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
        #save the model:
        save_original_transe(transe_model, 'transe_neworg_models')

    elif select_train_model == '2':
        transe_model = load_original_transe()
        #Create the optimizer:
        optimizer = torch.optim.SGD(transe_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

        #setup to train and evaluate the model:
        transe_model_train_eva = TransE_train_evaluate(train_dataset=train_dataset,
                                                       val_dataset=val_dataset,
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
        #save model
        save_original_transe(transe_model, 'transe_conorg_models')

    elif select_train_model == '3':
        select_eva_model = input('''Select model to be evaluated:\n
                                  1) original model\n
                                  2) continue training original model\n
                                  3) boosted model\n
                                  Enter (1,2,3):''')
        select_eva_model_num = input('''Enter the model number: ''')
        if select_eva_model == '1':
            evaluate_model = 'transe_neworg_models/transe_org_model_'
        elif select_eva_model == '2':
            evaluate_model = 'transe_conorg_models/transe_org_model_'
        elif select_eva_model == '3':
            evaluate_model = 'transe_boosted_models/transe_boost_model_'
        evaluate_model+=str(select_eva_model_num)+'.pt'
        print('Loading a TransE model from disk...')
        try:
            transe_model = torch.load(evaluate_model)
        except:
            print('Error in loading model. Aborting.')
            return -1
        print('Done!\nModel being evaluated: ', evaluate_model)
        transe_model_train_eva = TransE_train_evaluate(train_dataset=train_dataset,
                                                       val_dataset=val_dataset,
                                                       batch_size=None,
                                                       num_entity=num_entity,
                                                       model=transe_model,
                                                       device=device,
                                                       optimizer=None,
                                                       epoch=None,
                                                       seed=seed)
        # Evaluate TransE model:
        transe_model_train_eva.evaluate_model()

    elif select_train_model == '4':
        transe_model, start_epoch = restore_boosted_model()
        if start_epoch >= end_epoch:
            print('Already trained till end epoch condition')
            return 0
        #Create the optimizer:
        optimizer = torch.optim.SGD(transe_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
        # setup to train the model:
        test_obj = TransEBoost(start_epoch=start_epoch,
                               end_epoch=end_epoch,
                               train_data=train_dataset,
                               val_data=val_dataset,
                               seed=seed,
                               device=device,
                               batch_size=batch_size,
                               model=transe_model,
                               optimizer=optimizer,
                               num_entity=num_entity)
        #train the model:
        test_obj.train()


def restore_boosted_model():
    file_list=listdir('transe_boosted_models/')
    if len(file_list) == 0:
        print('Creating a new TransE model: ')
        transe_model = TransE(device=device,
                              num_entity=num_entity,
                              num_relation=num_relation,
                              emb_dim=emb_dim,
                              gamma=gamma,
                              seed=seed)
        print('Done!!')
        start_epoch=1
    else:
        print('Loading last trained model...')
        file_numbers=[int(re.findall('\d+', i)[0]) for i in file_list]
        model_num=max(file_numbers)
        transe_model=torch.load('transe_boosted_models/transe_boost_model_'+str(model_num)+'.pt')
        start_epoch=model_num+1
        print('Done!')
    return transe_model, start_epoch

def save_original_transe(model, prefix):
    file_list=listdir(prefix+'/')
    if len(file_list) == 0:
        torch.save(model, prefix+'/transe_org_model_1.pt')
    else:
        file_numbers=[int(re.findall('\d+', i)[0]) for i in file_list]
        model_num=max(file_numbers)
        model_num+=1
        torch.save(model, prefix+'/transe_org_model_'+str(model_num)+'.pt')
    print('Saving done !!')

def load_original_transe():
    file_list=listdir('transe_conorg_models/')
    if len(file_list) == 0:
        print('No existing model found')
        print('Creating a new TransE model: ')
        transe_model = TransE(device=device,
                              num_entity=num_entity,
                              num_relation=num_relation,
                              emb_dim=emb_dim,
                              gamma=gamma,
                              seed=seed)
        print('Done!!')
    else:
        file_numbers=[int(re.findall('\d+', i)[0]) for i in file_list]
        model_num=max(file_numbers)
        transe_model=torch.load('transe_conorg_models/transe_org_model_'+str(model_num)+'.pt')
    return transe_model

if __name__ == '__main__':
    main()