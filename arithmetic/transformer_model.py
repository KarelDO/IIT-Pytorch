import torch
from transformers import BertForSequenceClassification, BertTokenizer

class TransformerArithmetic(torch.nn.Module):
    # implement the indexing functions that get hooked to the model(coordinate system)
    # NOTE: possibility exists that we don't even need to implement this class. Just implement a new interventionable

    # we need to be able to:
    # do forward,
    # do interventions ( see interventionableTransformer )
    # get intermediate representations
    
    def __init__(self, config):
        super().__init__()

        self.config = dict(config)

        self.tokenizer = BertTokenizer.from_pretrained(self.config['from_pretrained'])
        # TODO: allow to specify different model but still copy most of the weights
        self.transformer = BertForSequenceClassification.from_pretrained(self.config['from_pretrained'])

        # extra classifier for the auxiliary task
        self.act_e = torch.nn.Tanh() if self.config['activation'] == "tanh" else torch.nn.LeakyReLU()
        self.ff_e = torch.nn.Linear(self.config['hidden_size'],3*self.config['dataset_highest_number'])

        self.act_s = torch.nn.Tanh() if self.config['activation'] == "tanh" else torch.nn.LeakyReLU()
        self.ff_s = torch.nn.Linear(self.config['hidden_size'],4*self.config['dataset_highest_number'])


    def forward(self, input):
        # transform the input to sentences with padding tokens
        sentences = []
        for i in input:
            sentences.append("{0} {1} {2} {3}".format(*i.numpy()))
        input = self.tokenizer(sentences, return_tensors='pt')

        output = self.transformer(**input, output_hidden_states=True)
        logits, hidden_states = output[0], output[1]
        # logits shape [bach_size / 2, num_labels]
        # hidden_states tuple each element [bach_size / 2, seq_len, hidden_dim]

        # NOTE: this is because default bert has two labels, change this in the config
        pooler_output = hidden_states[5][:,0]
        pooler_output = self.act_s(self.ff_s(pooler_output))

        # NOTE: would be nice to use same indexing function as the one specified for the hooks
        # NOTE: maybe we should just mimic the pooling architecture, to follow best practises
        e_state = hidden_states[4][:,0]
        e = self.act_e(self.ff_e(e_state))

        return pooler_output, e
        