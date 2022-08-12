import json
import torch

from train_and_evaluate import TransETrain, Evaluation
from transe_boost import TransEBoost, TransEBoost2
from data_handling import LoadMetaDataHandling, Draw


def main():
    folder_list = ['transe_original', 'transe_selftrain_type_1', 'transe_selftrain_type_2']
    dataset_num_map = {'FB15k': '1', 'FB15k237': '2', 'WN18': '3'}
    load_data = LoadMetaDataHandling(folder_list=folder_list, dataset_num_map=dataset_num_map)

    select_option = input('''
                               \n1) Train a basic TransE model? 
                               \n2) Evaluate a model? 
                               \n3) Training a self-training type 1 model?
                               \n4) Training a self-training type 2 model?
                               \n5) Draw graphs for evaluated metrics?
                               \n6) Exit (any other input will be lead to exit)
                               \n Enter (1, 2, 3, 4, 5): ''')

    if select_option == '1':
        # Get the transe model:
        meta_data, exp_dir_name, transe_model, train_dataset, val_dataset, test_dataset = load_data.load(folder_list[0])
        # Create the optimizer:
        optimizer = torch.optim.SGD(transe_model['cur_model'].parameters(),
                                    lr=meta_data['global']['lr'],
                                    weight_decay=meta_data['global']['l2'])
        # Train the model:
        transe_model_train = TransETrain(train_dataset=train_dataset,
                                         batch_size=meta_data['global']['batch size'],
                                         num_entity=meta_data['global']['num entity'],
                                         model=transe_model['cur_model'],
                                         device=meta_data['global']['device'],
                                         optimizer=optimizer,
                                         start_epoch=meta_data['global']['latest epoch'] + 1,
                                         end_epoch=meta_data['global']['total epoch'],
                                         seed=meta_data['global']['seed'],
                                         folder=exp_dir_name)
        # train TransE model:
        transe_model_train.train()

    elif select_option == '2':
        evaluate_model = load_data.select_folder()
        if evaluate_model == -1:
            return -1
        select_exp_num = load_data.select_exp(folder=evaluate_model)
        evaluate_model += '/' + 'exp_' + select_exp_num + '/'
        select_eva_model_num = load_data.select_model_num(exp_dir_name=evaluate_model)
        with open(evaluate_model + 'meta_data.json', 'r+') as json_file:
            meta_data = json.load(json_file)
            # get the appropriate dataset to evaluate the model
            automatic_input = load_data.dataset_name_to_number(dataset_name=meta_data['global']['dataset name'])
            load_data.choose_dataset(automatic_input=automatic_input)
            train_dataset, val_dataset, test_dataset = load_data.get_dataset()
            for eva_model_num in range(select_eva_model_num[0], select_eva_model_num[1] + 1):
                eva_model_num = str(eva_model_num)
                # load the model to be evaluated
                print('Loading a TransE model from disk...')
                try:
                    transe_model = torch.load(evaluate_model + 'transe_' + eva_model_num + '.pt')
                except:
                    print('Error in loading model. Aborting.')
                    return -1
                print('Done!\nModel being evaluated: ', evaluate_model + 'transe_' + eva_model_num + '.pt')
                # specify the values for evaluation:
                eva_obj = Evaluation(data=val_dataset,
                                     model=transe_model['cur_model'],
                                     num_entity=meta_data['global']['num entity'],
                                     device=meta_data['global']['device'])
                # Evaluate TransE model:
                mr, mrr, hits_at_10 = eva_obj.evaluate_model()
                # store the values in meta data file for that model
                meta_data['local']['transe_' + eva_model_num]['MR'] = float(mr)
                meta_data['local']['transe_' + eva_model_num]['MRR'] = float(mrr)
                meta_data['local']['transe_' + eva_model_num]['Hits@10'] = float(hits_at_10)
            json_file.seek(0)
            json.dump(meta_data, json_file, indent=4)
            json_file.truncate()
            json_file.close()

    elif select_option == '3':
        meta_data, exp_dir_name, transe_model, train_dataset, val_dataset, test_dataset = load_data.load(folder_list[1])
        # Create the optimizer:
        optimizer = torch.optim.SGD(transe_model['cur_model'].parameters(),
                                    lr=meta_data['global']['lr'],
                                    weight_decay=meta_data['global']['l2'])
        # setup to train the model:
        self_train_type1 = TransEBoost(start_epoch=meta_data['global']['latest epoch'] + 1,
                                       end_epoch=meta_data['global']['total epoch'],
                                       train_data=train_dataset,
                                       seed=meta_data['global']['seed'],
                                       device=meta_data['global']['device'],
                                       batch_size=meta_data['global']['batch size'],
                                       model=transe_model['cur_model'],
                                       optimizer=optimizer,
                                       num_entity=meta_data['global']['num entity'],
                                       folder=exp_dir_name)
        # train the model:
        self_train_type1.train()

    elif select_option == '4':
        meta_data, exp_dir_name, transe_model, train_dataset, val_dataset, test_dataset = load_data.load(folder_list[2])
        # Check if transe_model is a dict or not:
        if str(type(transe_model)) != "<class 'dict'>":
            temp = transe_model
            transe_model = {'pre_model': temp,
                            'cur_model': temp
                            }
        elif 'pre_model' not in  transe_model.keys():
            transe_model['pre_model'] = transe_model['cur_model']
        if meta_data['global']['latest epoch'] == 0:
            num_model_train = int(input('Enter number of models to train: '))
            meta_data = load_data.meta_data_add_field(exp_dir_name=exp_dir_name,
                                                      # cur_model_num=1,
                                                      num_model_train=num_model_train)
            start_epoch = 1
            start_model = 1
        else:
            start_epoch = meta_data['global']['latest epoch'] % meta_data['global']['total epoch']
            start_model = int(meta_data['global']['latest epoch'] / meta_data['global']['total epoch']) + 1
            if start_epoch == 0:
                start_epoch = 1
            else:
                start_epoch += 1
        # Create the optimizer:
        optimizer = torch.optim.SGD(transe_model['cur_model'].parameters(),
                                    lr=meta_data['global']['lr'],
                                    weight_decay=meta_data['global']['l2'])
        # setup the transe boost 2
        boost2_obj = TransEBoost2(pre_model=transe_model['pre_model'],
                                  cur_model=transe_model['cur_model'],
                                  train_data=train_dataset,
                                  start_epoch=start_epoch,
                                  end_epoch=meta_data['global']['total epoch'],
                                  total_epoch=meta_data['global']['latest epoch'],
                                  seed=meta_data['global']['seed'],
                                  device=meta_data['global']['device'],
                                  batch_size=meta_data['global']['batch size'],
                                  start_model=start_model,
                                  end_model=meta_data['global']['num_model_train'],
                                  optimizer=optimizer,
                                  num_entity=meta_data['global']['num entity'],
                                  folder=exp_dir_name)
        # train:
        boost2_obj.train()

    elif select_option == '5':
        draw_obj = Draw()
        num_model_compare = int(input('Enter number of models to display in a graph: '))
        draw_model_met_dict = {}
        for model_num in range(num_model_compare):
            folder_name = load_data.select_folder()
            exp_num = load_data.select_exp(folder=folder_name)
            exp_dir_name = folder_name + '/' + 'exp_' + str(exp_num) + '/'
            model_name = input('Enter model name: ')
            draw_model_met_dict[model_name] = exp_dir_name
        draw_obj.plot_mr(mr_dict=draw_model_met_dict, title='')
        draw_obj.plot_mrr(mrr_dict=draw_model_met_dict, title='')
        draw_obj.plot_hits(hits_dict=draw_model_met_dict, title='')
        draw_obj.plot_tr_loss(tr_dict=draw_model_met_dict, title='')

    elif select_option == '6':
        pass

    else:
        return 0


if __name__ == '__main__':
    main()