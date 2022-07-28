import re
import torch
from pykeen.datasets import FB15k237, FB15k, WN18
from models import TransE
from train_and_evaluate import TransE_train_evaluate
# from boost import TransEBoost
from transe_boost import TransEBoost
from os import listdir
import json

#set transe model paratmeters:
device='cuda'
emb_dim=50
gamma=1

#set transe optimizer parameters:
lr=0.01
weight_decay=0

#transe training parameters:
batch_size=36
epoch=2

#set torch seeds:
seed=2022
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#TransEBoost parameters:
end_epoch=100

def main():

    select_dataset = input('''1)FB15K\n2)FB15K237\n3)WN18\n4)Exit\nEnter 1,2,3 or 4:''')
    if select_dataset == '1':
        fb15k_dataset = FB15k()
        train_dataset = fb15k_dataset.training.mapped_triples
        val_dataset = fb15k_dataset.validation.mapped_triples
        test_dataset = fb15k_dataset.testing.mapped_triples
        num_entity = fb15k_dataset.num_entities.real
        num_relation = fb15k_dataset.num_relations.real
        dataset_name = 'FB15k'
    elif select_dataset == '2':
        fb15k237_dataset = FB15k237()
        train_dataset = fb15k237_dataset.training.mapped_triples
        val_dataset = fb15k237_dataset.validation.mapped_triples
        test_dataset = fb15k237_dataset.testing.mapped_triples
        num_entity = fb15k237_dataset.num_entities.real
        num_relation = fb15k237_dataset.num_relations.real
        dataset_name = 'FB15k237'
    elif select_dataset == '3':
        wn18_dataset = WN18()
        train_dataset = wn18_dataset.training.mapped_triples
        val_dataset = wn18_dataset.validation.mapped_triples
        test_dataset = wn18_dataset.testing.mapped_triples
        num_entity = wn18_dataset.num_entities.real
        num_relation = wn18_dataset.num_relations.real
        dataset_name = 'WN18'
    elif select_dataset == '4':
        return 0

    #print their shapes
    print('Training dataset size: ', train_dataset.shape)
    print('Validation dataset size: ', val_dataset.shape)
    print('Testing dataset size: ', test_dataset.shape)
    print('Number of Entities: ', num_entity)
    print('Number of relations: ', num_relation)

    select_train_model = input('''1) Train a new TransE model?
                               \n2) Train a existing TransE model? 
                               \n3) Evaluate a model? 
                               \n4) Resume Training a boosted model?
                               \n5) Exit 
                               \n Enter (1, 2, 3, 4, 5): ''')

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
        transe_model = load_original_transe(num_entity=num_entity, num_relation=num_relation)
        if transe_model == -1:
            return -1
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
        transe_model, start_epoch = restore_boosted_model(num_entity=num_entity, num_relation=num_relation)
        if transe_model == -1:
            return -1
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

    elif select_train_model == '5':
        return 0


def restore_boosted_model(num_entity, num_relation):
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
        try:
            transe_model=torch.load('transe_boosted_models/transe_boost_model_'+str(model_num)+'.pt')
        except:
            print('Error encountered while loading model. Aborting.')
            return -1, -1
        start_epoch=model_num+1
        print('Done!')
    return transe_model, start_epoch

def save_original_transe(model, folder):
    file_list=listdir(folder + '/')
    if len(file_list) == 0:
        torch.save(model, folder + '/transe_org_model_1.pt')
    else:
        file_numbers=[int(re.findall('\d+', i)[0]) for i in file_list]
        model_num=max(file_numbers)
        model_num+=1
        torch.save(model, folder + '/transe_org_model_' + str(model_num) + '.pt')
    print('Saving done !!')

def load_original_transe(num_entity, num_relation):
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
        try:
            transe_model=torch.load('transe_conorg_models/transe_org_model_'+str(model_num)+'.pt')
        except:
            print('Error encountered while loading model. Aborting.')
            return -1
    return transe_model

if __name__ == '__main__':
    main()