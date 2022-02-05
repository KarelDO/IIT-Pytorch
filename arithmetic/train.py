import torch
import random
import math
from dataset import ArithmeticDataset
from causal_model import CausalArithmetic
from neural_model import NeuralArithmetic
from interventionable import Interventionable

###
from torchviz import make_dot


###
def ii_accuracy(neural_model, causal_model, alignment, ds):
    neural_model.model.eval()

    neural_node, causal_node = alignment

    dl = iter(torch.utils.data.DataLoader(ds, batch_size = 64, shuffle = False))

    correct = 0
    for i in range(len(dl)):
        x, y = dl.next()

        # get source and base
        halfway_point = math.floor(x.shape[0]/2) 

        x_base, y_base = x[:halfway_point], y[:halfway_point]
        x_source, y_source = x[halfway_point:], y[halfway_point:]
    
        with torch.no_grad():
            _, _, predict_intervention = neural_model.forward(x_source, x_base, neural_node)
            _, _, target_intervention = causal_model.forward(x_source, x_base, causal_node)

            predict_labels = torch.argmax(predict_intervention, dim=1)
    
            correct += sum(predict_labels == target_intervention)

    acc = 50 * correct / len(ds)

    neural_model.model.train()

    return correct, acc

def eval(model, ds):
    model.model.eval()

    task_criterion = torch.nn.CrossEntropyLoss()
    
    predict = model.model(ds.x)
    loss = task_criterion(predict, ds.y)

    model.model.train()

    return loss


def train_with_interventions(neural_model, causal_model, alignments, ds_train, ds_test):
    neural_model.model.train()

    N_EPOCHS = 500

    LR = 0.001
    MOMENTUM = 0.9

    task_criterion = torch.nn.CrossEntropyLoss()
    iit_criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(neural_model.model.parameters(), lr=LR, momentum=MOMENTUM)

    running_task_loss = 0.0
    running_iit_loss = 0.0

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(N_EPOCHS):
        dl = iter(torch.utils.data.DataLoader(ds_train, batch_size = 64, shuffle = True))
        for i in range(len(dl)):
            x, y = dl.next()

            # get source and base
            halfway_point = math.floor(x.shape[0]/2) 

            x_base, y_base = x[:halfway_point], y[:halfway_point]
            x_source, y_source = x[halfway_point:], y[halfway_point:]


            # run model on base
            predict = neural_model.model(x_base)
            task_loss = task_criterion(predict, y_base)

            optimizer.zero_grad()
            task_loss.backward()
            optimizer.step()

            # run intervention
            neural_node, causal_node = random.choice(alignments)
            _, _, predict_intervention = neural_model.forward(x_source, x_base, neural_node)
            with torch.no_grad():
                _, _, target_intervention = causal_model.forward(x_source, x_base, causal_node)
            iit_loss = iit_criterion(predict_intervention, target_intervention)

            # make_dot(base, params=dict(list(neural_model.model.named_parameters()))).render("./base", format="png")  
            # make_dot(predict_intervention, params=dict(list(neural_model.model.named_parameters()))).render("./tmp_counterfactual_copy", format="png")  
            # exit

            optimizer.zero_grad()
            iit_loss.backward()
            optimizer.step()

            running_task_loss += task_loss.item()
            running_iit_loss += iit_loss.item()

        print(f'[epoch {epoch + 1}] task loss: {running_task_loss / (len(dl)/2):.3f}') 
        print(f'[epoch {epoch + 1}] iit loss: {running_iit_loss / (len(dl)/2):.3f}') 
        running_task_loss = 0.0
        running_iit_loss = 0.0

        if epoch % 100 == 0:
            test_task_loss = eval(neural_model,ds_test)
            print(f'[epoch {epoch + 1}] test task loss: {test_task_loss:.3f}') 

            for alignment in alignments:
                correct, acc = ii_accuracy(neural_model, causal_model, alignment, ds_test)
                print(f'[epoch {epoch + 1}] ii accuracy {alignment}: {acc:.3f}') 


            

if __name__ == "__main__":
    highest_number = 30
    hidden_width = 100

    train_size = 10000
    test_size = 1000

    ds_train = ArithmeticDataset(amount=train_size, high=highest_number)
    ds_test = ArithmeticDataset(amount=test_size, high=highest_number)
    
    neural_model = Interventionable(NeuralArithmetic(highest_number,hidden_width))
    causal_model = Interventionable(CausalArithmetic())
    
    # print(list(neural_model.model.named_children()))
    # print(list(causal_model.model.named_children()))

    # neural -> causal
    alignments = [
        ('identity_x', 'x'),
        ('identity_y', 'y'),
        ('identity_z', 'z'),
        ('identity_d', 'S'),
        ('identity_o', 'O')
    ]
    train_with_interventions(neural_model, causal_model, alignments, ds_train, ds_test)

    # eval(model, ds_test)

