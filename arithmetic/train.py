from re import M
import torch
import random
import numpy as np
import math
from dataset import ArithmeticDataset
from causal_model import CausalArithmetic
from neural_model import NeuralArithmetic
from interventionable import Interventionable
import wandb
import pyhocon
import time
import sys


def ii_accuracy(neural_model, causal_model, alignment, ds, config):
    neural_model.model.eval()

    neural_node, causal_node = alignment

    dl = iter(torch.utils.data.DataLoader(
        ds, batch_size=config['batch_size'], shuffle=False))

    correct = 0
    for i in range(len(dl)):
        x, y = dl.next()
        x, y = x.to(config['device']), y.to(config['device'])
        # get source and base
        halfway_point = math.floor(x.shape[0]/2)

        x_base, y_base = x[:halfway_point], y[:halfway_point]
        x_source, y_source = x[halfway_point:], y[halfway_point:]

        with torch.no_grad():
            _, _, predict_intervention = neural_model.forward(
                x_source, x_base, neural_node)
            _, _, target_intervention = causal_model.forward(
                x_source, x_base, causal_node)

            predict_labels = torch.argmax(predict_intervention, dim=1)

            correct += sum(predict_labels == target_intervention)

    acc = 50 * correct / len(ds)

    neural_model.model.train()

    return correct, acc


def eval(model, ds, config):
    model.model.eval()

    task_criterion = torch.nn.CrossEntropyLoss()

    predict = model.model(ds.x.to(config['device']))
    loss = task_criterion(predict, ds.y.to(config['device']))

    model.model.train()

    return loss, torch.argmax(predict, dim=1)


def train_log(epoch, len_ds, running_task_loss, running_iit_loss):
    task_loss = running_task_loss / (len_ds)
    iit_loss = running_iit_loss / (len_ds/2)
    wandb.log({"train loss": task_loss, "iit loss": iit_loss}, step=epoch)
    print(f'[epoch {epoch + 1}] train loss: {task_loss:.3f}')
    print(f'[epoch {epoch + 1}] iit loss: {iit_loss:.3f}')


def eval_log(epoch, ds_test, neural_model, causal_model, config):
    test_task_loss, prediction = eval(neural_model, ds_test, config)

    wandb.log({"test loss": test_task_loss}, step=epoch)
    print(f'[epoch {epoch + 1}] test loss: {test_task_loss:.3f}')

    for alignment in config['alignments']:
        correct, acc = ii_accuracy(
            neural_model, causal_model, alignment, ds_test, config)
        wandb.log({f"ii  accuracy {alignment}": acc}, step=epoch)
        print(f'[epoch {epoch + 1}] ii accuracy {alignment}: {acc:.3f}')

    test_data = list(zip(ds_test.x, ds_test.y, prediction))
    test_columns = ["input", "true label", "predicted label"]
    test_table = wandb.Table(data=test_data, columns=test_columns)
    wandb.log({"test_table": test_table}, step=epoch)

    error_data = prediction - ds_test.y.to(config['device'])
    wandb.log({"error_histogram": wandb.Histogram(error_data.cpu())}, step=epoch)

    accuracy = sum(prediction == ds_test.y.to(config['device']))/len(ds_test.y.to(config['device']))
    wandb.log({"test accuracy": accuracy}, step=epoch)


def prepare_training(config):
    # create causal and neural model
    neural_model = Interventionable(NeuralArithmetic(config).to(config['device']))
    causal_model = Interventionable(CausalArithmetic(config).to(config['device']))

    print(neural_model.model)

    # create test and train set
    ds_train = ArithmeticDataset(
        size=config['dataset_train_size'], highest_number=config['dataset_highest_number'])
    ds_test = ArithmeticDataset(
        size=config['dataset_test_size'], highest_number=config['dataset_highest_number'])

    # criterions
    task_criterion = torch.nn.CrossEntropyLoss()
    iit_criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(neural_model.model.parameters(
    ), lr=config['learning_rate'], momentum=config['momentum'])

    return neural_model, causal_model, ds_train, ds_test, task_criterion, iit_criterion, optimizer


def train_with_interventions(neural_model, causal_model, ds_train, ds_test, task_criterion, iit_criterion, optimizer, config):
    neural_model.model.train()

    wandb.watch(neural_model.model, task_criterion,
                log="all", log_freq=config['eval_freq'])

    running_task_loss = 0.0
    running_iit_loss = 0.0

    t1 = time.time()

    for epoch in range(config['epochs']):
        dl = iter(torch.utils.data.DataLoader(
            ds_train, batch_size=config['batch_size'], shuffle=True))
        for i in range(len(dl)):
            x, y = dl.next()
            x, y = x.to(config['device']), y.to(config['device'])

            # get source and base examples
            # NOTE: ignoring the impactfulness of interventions for now
            halfway_point = math.floor(x.shape[0]/2)

            x_base, y_base = x[:halfway_point], y[:halfway_point]
            x_source, y_source = x[halfway_point:], y[halfway_point:]

            # run intervention
            neural_node, causal_node = random.choice(config['alignments'])
            source_logits, base_logits, counterfactual_logits = neural_model.forward(
                x_source, x_base, neural_node)
            with torch.no_grad():
                _, _, counterfactual_target = causal_model.forward(
                    x_source, x_base, causal_node)

            # task loss on all seen examples
            task_loss = task_criterion(torch.cat(
                (source_logits, base_logits), dim=0), torch.cat((y_source, y_base), dim=0))
            # iit loss
            iit_loss = iit_criterion(
                counterfactual_logits, counterfactual_target)

            # step
            optimizer.zero_grad()
            loss = task_loss + iit_loss
            loss.backward()
            optimizer.step()

            running_task_loss += task_loss.item()
            running_iit_loss += iit_loss.item()

        train_log(epoch, len(ds_train), running_task_loss, running_iit_loss)

        running_task_loss = 0.0
        running_iit_loss = 0.0

        if epoch % config['eval_freq'] == 0:
            wandb.log({"time": time.time() - t1},step=epoch)
            eval_log(epoch, ds_test, neural_model,
                     causal_model, config)

    torch.onnx.export(neural_model.model, x,
                      "neural_model.onnx", opset_version=11)
    torch.onnx.export(causal_model.model, x,
                      "causal_model.onnx", opset_version=11)
    wandb.save("neural_model.onnx")
    wandb.save("causal_model.onnx")


def create_config(experiment_name):
    config = pyhocon.ConfigFactory.parse_file(
        "experiments.conf")[experiment_name]
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
    wandb.login()

    config = create_config(sys.argv[1])
    set_seeds(config)

    neural_model, causal_model, ds_train, ds_test, task_criterion, iit_criterion, optimizer = prepare_training(
        config)

    with wandb.init(project='causal-arithmetic', entity='stanford-causality', config=config):
        config = wandb.config
        train_with_interventions(
            neural_model, causal_model, ds_train, ds_test, task_criterion, iit_criterion, optimizer, config)
