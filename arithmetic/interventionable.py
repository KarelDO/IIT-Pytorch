import torch

class Interventionable():
    def __init__(self, model):
        self.activation = {}
        self.model = model

        self.names_to_layers = dict(self.model.named_children())

    def _get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook

    def _set_activation(self, name):
        def hook(model, input, output):
            return self.activation[name]
        return hook

    def forward(self, source, base, layer_name):
        # NOTE: other ways that do not require constantly adding / removing hooks should exist
        assert source.shape == base.shape
        assert layer_name in self.names_to_layers

        # set hook to get activation
        get_handler = self.names_to_layers[layer_name].register_forward_hook(self._get_activation(layer_name))

        # get output on source examples (and also capture the activations)
        # with torch.no_grad():
        #     source_logits = self.model(source)
        source_logits = self.model(source)

        # remove the handler (don't store activations on base) 
        get_handler.remove()

        # get base logits
        base_logits = self.model(base)
        
        # set hook to do the intervention
        set_handler = self.names_to_layers[layer_name].register_forward_hook(self._set_activation(layer_name))

        # get counterfactual output on base examples
        counterfactual_logits = self.model(base)

        # remove the handler
        set_handler.remove()

        return source_logits, base_logits, counterfactual_logits

