from re import M
import torch
import random
import numpy as np
import math
from dataset import ArithmeticDataset2
from causal_model import CausalArithmetic2
from transformer_model import TransformerArithmetic
from interventionable import InterventionableTransformer, Interventionable2
from test_dataset import check_overlap
import wandb
import pyhocon
import time
import sys
import math
from transformers import AdamW, get_scheduler


def ii_accuracy(neural_model, causal_model, alignment, ds, config, task='1'):
    neural_model.model.eval()

    neural_node, causal_node = alignment

    dl = iter(torch.utils.data.DataLoader(
        ds, batch_size=config['batch_size'], shuffle=False))

    correct = 0
    for i in range(len(dl)):
        x, _, _ = dl.next()
        x = x.to(config['device'])
        # get source and base
        halfway_point = math.floor(x.shape[0]/2)

        x_base = x[:halfway_point]
        x_source = x[halfway_point:]

        with torch.no_grad():
            if task == '1':
                _, _, predict_intervention, _, _, _ = neural_model.forward(
                    x_source, x_base, neural_node)
                _, _, target_intervention, _, _, _ = causal_model.forward(
                    x_source, x_base, causal_node)
            if task == '2':
                _, _, _, _, _, predict_intervention = neural_model.forward(
                    x_source, x_base, neural_node)
                _, _, _, _, _, target_intervention = causal_model.forward(
                    x_source, x_base, causal_node)

            predict_labels = torch.argmax(predict_intervention, dim=1)

            correct += sum(predict_labels == target_intervention)

    acc = 100 * correct / (len(ds)/2)

    neural_model.model.train()

    return correct, acc


def eval(model, ds, config):
    model.model.eval()

    task_criterion = torch.nn.CrossEntropyLoss()

    predict_T1, predict_T2 = model.model(ds.x.to(config['device']))
    loss_T1 = task_criterion(predict_T1, ds.y_T1.to(config['device']))
    loss_T2 = task_criterion(predict_T2, ds.y_T2.to(config['device']))

    model.model.train()

    return loss_T1, torch.argmax(predict_T1, dim=1), loss_T2, torch.argmax(predict_T2, dim=1)


def train_log(epoch, len_ds, running_T1_task_loss, running_T2_task_loss, running_iit_loss, config):
    T1_task_loss = running_T1_task_loss / (len_ds)
    T2_task_loss = running_T2_task_loss / (len_ds)
    iit_loss = running_iit_loss / (len_ds/2)
    
    if 'wandb_track' in config and config['wandb_track'] == 'True':
        wandb.log({"T1 train loss": T1_task_loss}, step=epoch)
        wandb.log({"T2 train loss": T2_task_loss}, step=epoch)
        wandb.log({"iit loss": iit_loss}, step=epoch)
    print(f'[epoch {epoch + 1}] T1 train loss: {T1_task_loss:.3f}')
    print(f'[epoch {epoch + 1}] T2 train loss: {T2_task_loss:.3f}')
    print(f'[epoch {epoch + 1}] iit loss: {iit_loss:.3f}')


def eval_log(epoch, ds_test, neural_model, causal_model, config):
    test_loss_T1, prediction_T1, test_loss_T2, prediction_T2 = eval(
        neural_model, ds_test, config)

    #wandb.log({"test loss": test_task_loss}, step=epoch)
    print(f'[epoch {epoch + 1}] T1 test loss: {test_loss_T1:.3f}')
    print(f'[epoch {epoch + 1}] T2 test loss: {test_loss_T2:.3f}')

    # TODO: eval on second alignments
    for alignment in config['alignments1']:
        correct, acc = ii_accuracy(
            neural_model, causal_model, alignment, ds_test, config, task='1')
        if 'wandb_track' in config and config['wandb_track'] == 'True':
            wandb.log({f"T1 ii  accuracy {alignment}": acc}, step=epoch)
        print(f'[epoch {epoch + 1}] T1 ii accuracy {alignment}: {acc:.3f}')

    for alignment in config['alignments2']:
        correct, acc = ii_accuracy(
            neural_model, causal_model, alignment, ds_test, config, task='2')
        if 'wandb_track' in config and config['wandb_track'] == 'True':
            wandb.log({f"T2 ii  accuracy {alignment}": acc}, step=epoch)
        print(f'[epoch {epoch + 1}] T2 ii accuracy {alignment}: {acc:.3f}')

    # test_data = list(zip(ds_test.x, ds_test.y, prediction))
    # test_columns = ["input", "true label", "predicted label"]
    # test_table = wandb.Table(data=test_data, columns=test_columns)
    #wandb.log({"test_table": test_table}, step=epoch)

    error_data_T1 = prediction_T1 - ds_test.y_T1.to(config['device'])
    error_data_T2 = prediction_T2 - ds_test.y_T2.to(config['device'])
    if 'wandb_track' in config and config['wandb_track'] == 'True':
        wandb.log({"T1 error_histogram": wandb.Histogram(error_data_T1.cpu())}, step=epoch)
        wandb.log({"T2 error_histogram": wandb.Histogram(error_data_T2.cpu())}, step=epoch)

    accuracy_T1 = sum(prediction_T1 == ds_test.y_T1.to(
        config['device']))/len(ds_test.y_T1.to(config['device']))
    accuracy_T2 = sum(prediction_T2 == ds_test.y_T2.to(
        config['device']))/len(ds_test.y_T2.to(config['device']))
    if 'wandb_track' in config and config['wandb_track'] == 'True':
        wandb.log({"T1 test accuracy": accuracy_T1}, step=epoch)
        wandb.log({"T2 test accuracy": accuracy_T2}, step=epoch)


def prepare_training(config):
    # create causal and neural model
    neural_model = InterventionableTransformer(
        TransformerArithmetic(config).to(config['device']))
    causal_model = Interventionable2(
        CausalArithmetic2(config).to(config['device']))

    print(neural_model.model)

    # create test and train set
    ds_train = ArithmeticDataset2(
        size=config['dataset_train_size'], highest_number=config['dataset_highest_number'], seed=config['seed'])
    ds_test = ArithmeticDataset2(
        size=config['dataset_test_size'], highest_number=config['dataset_highest_number'], seed=config['seed']+1)

    print("database overlap: ",check_overlap(ds_train, ds_test) )

    # criterions
    task_criterion = torch.nn.CrossEntropyLoss()
    iit_criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = AdamW(neural_model.model.parameters(), lr=config['learning_rate'])

    # scheduler
    num_training_steps = int(math.floor(config['epochs'] * len(ds_train) / config['batch_size']))
    num_warmup_steps = config['num_warmup_steps']
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


    return neural_model, causal_model, ds_train, ds_test, task_criterion, iit_criterion, optimizer


def train_with_interventions(neural_model, causal_model, ds_train, ds_test, task_criterion, iit_criterion, optimizer, config):
    neural_model.model.train()

    if 'wandb_track' in config and config['wandb_track'] == 'True':
        wandb.watch(neural_model.model, task_criterion, log="all", log_freq=config['eval_freq'])

    running_T1_task_loss = 0.0
    running_T2_task_loss = 0.0
    running_iit_loss = 0.0

    t1 = time.time()

    for epoch in range(config['epochs']):
        dl = iter(torch.utils.data.DataLoader(
            ds_train, batch_size=config['batch_size'], shuffle=True))
        for i in range(len(dl)):
            x, y_T1, y_T2 = dl.next()
            x, y_T1, y_T2 = x.to(config['device']), y_T1.to(
                config['device']), y_T2.to(config['device'])
            # TODO: tokenization

            # get source and base examples
            # NOTE: ignoring the impactfulness of interventions for now
            halfway_point = math.floor(x.shape[0]/2)

            x_base, y_T1_base, y_T2_base = x[:halfway_point], y_T1[:halfway_point], y_T2[:halfway_point]
            x_source, y_T1_source, y_T2_source = x[halfway_point:
                                                   ], y_T1[halfway_point:], y_T2[halfway_point:]

            # run intervention
            # TODO: only run interventions for an auxiliary task
            neural_node, causal_node = random.choice(config['alignments2'])
            source_logits_T1, base_logits_T1, _, source_logits_T2, base_logits_T2, counterfactual_logits_T2 = neural_model.forward(
                x_source, x_base, neural_node)
            with torch.no_grad():
                _, _, _, _, _, counterfactual_target_T2 = causal_model.forward(
                    x_source, x_base, causal_node)

            # task loss on all seen examples
            if 'T1_delay' in config and epoch < config['T1_delay']:
                T1_task_loss = torch.zeros((1,), device=config['device'])
            else:
                T1_task_loss = task_criterion(torch.cat(
                    (source_logits_T1, base_logits_T1), dim=0), torch.cat((y_T1_source, y_T1_base), dim=0))

            if 'T1_warmup' in config and config['T1_delay'] and epoch >= config['T1_delay']:
                warmup = min((epoch-config['T1_delay'])/config['T1_warmup'],1.0)
            else:
                warmup = 1.0
            T1_task_loss *= warmup
            
            T2_task_loss = task_criterion(torch.cat(
                (source_logits_T2, base_logits_T2), dim=0), torch.cat((y_T2_source, y_T2_base), dim=0))
            # iit loss
            if config['iit'] == 'True':
                iit_loss = iit_criterion(
                    counterfactual_logits_T2, counterfactual_target_T2)
            else:
                iit_loss = torch.zeros((1,), device=config['device'])

            # regularization
            if 'reg' in config and config['reg'] == 'l1':
                reg_loss = sum(p.abs().sum() for p in neural_model.model.parameters())
                reg_loss *= config['reg_alpha']
            else:
                reg_loss = torch.zeros((1,), device=config['device'])
            
            # step
            optimizer.zero_grad()
            loss = T1_task_loss + T2_task_loss + iit_loss + reg_loss
            loss.backward()
            optimizer.step()

            running_T1_task_loss += T1_task_loss.item()
            running_T2_task_loss += T2_task_loss.item()
            running_iit_loss += iit_loss.item()

        train_log(epoch, len(ds_train), running_T1_task_loss, running_T2_task_loss, running_iit_loss, config)

        running_T1_task_loss = 0.0
        running_T2_task_loss = 0.0
        running_iit_loss = 0.0

        if epoch % config['eval_freq'] == 0:
            if 'wandb_track' in config and config['wandb_track'] == 'True':
                wandb.log({"time": time.time() - t1},step=epoch)
            eval_log(epoch, ds_test, neural_model,
                     causal_model, config)

    torch.onnx.export(neural_model.model, x,
                      "neural_model.onnx", opset_version=11)
    torch.onnx.export(causal_model.model, x,
                      "causal_model.onnx", opset_version=11)
    if 'wandb_track' in config and config['wandb_track'] == 'True':
        wandb.save("neural_model.onnx")
        wandb.save("causal_model.onnx")


def create_config(experiment_name):
    config = pyhocon.ConfigFactory.parse_file(
        "./experiments3.conf")[experiment_name]
    config['device'] = torch.device(
        config['device'] if torch.cuda.is_available() else "cpu")

    return config


def set_seeds(config):
    torch.backends.cudnn.deterministic = True
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])


if __name__ == "__main__":

    # config = create_config(sys.argv[1])
    # TODO: change
    config = create_config('base')
    set_seeds(config)

    if 'wandb_track' in config and config['wandb_track'] == 'True':
        wandb.login()

    neural_model, causal_model, ds_train, ds_test, task_criterion, iit_criterion, optimizer = prepare_training(
        config)


    if 'wandb_track' in config and config['wandb_track'] == 'True':
        with wandb.init(project='transformer-arithmetic-multiIIT', entity='stanford-causality', config=config):
            config = wandb.config
            train_with_interventions(
                neural_model, causal_model, ds_train, ds_test, task_criterion, iit_criterion, optimizer, config)
    else:
        train_with_interventions(
            neural_model, causal_model, ds_train, ds_test, task_criterion, iit_criterion, optimizer, config)
