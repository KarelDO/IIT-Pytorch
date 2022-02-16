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
        get_handler = self.names_to_layers[layer_name].register_forward_hook(
            self._get_activation(layer_name))

        # get output on source examples (and also capture the activations)
        # with torch.no_grad():
        #     source_logits = self.model(source)
        source_logits = self.model(source)

        # remove the handler (don't store activations on base)
        get_handler.remove()

        # get base logits
        base_logits = self.model(base)

        # set hook to do the intervention
        set_handler = self.names_to_layers[layer_name].register_forward_hook(
            self._set_activation(layer_name))

        # get counterfactual output on base examples
        counterfactual_logits = self.model(base)

        # remove the handler
        set_handler.remove()

        return source_logits, base_logits, counterfactual_logits


class Interventionable2():
    # NOTE: can probably be merged with Interventionable1
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
        get_handler = self.names_to_layers[layer_name].register_forward_hook(
            self._get_activation(layer_name))

        # get output on source examples (and also capture the activations)
        # with torch.no_grad():
        #     source_logits = self.model(source)
        source_logits_T1, source_logits_T2 = self.model(source)

        # remove the handler (don't store activations on base)
        get_handler.remove()

        # get base logits
        base_logits_T1, base_logits_T2 = self.model(base)

        # set hook to do the intervention
        set_handler = self.names_to_layers[layer_name].register_forward_hook(
            self._set_activation(layer_name))

        # get counterfactual output on base examples
        counterfactual_logits_T1, counterfactual_logits_T2 = self.model(base)

        # remove the handler
        set_handler.remove()

        return source_logits_T1, base_logits_T1, counterfactual_logits_T1, source_logits_T2, base_logits_T2, counterfactual_logits_T2

class InterventionableTransformer():
    # NOTE: can probably be merged with Interventionable1
    def __init__(self, model):
        self.activation = {}
        self.model = model

    # these functions are model dependent
    # they specify how the coordinate system works
    def _coordinate_to_getter(self, coord):
        layer, index = coord
        def hook(model, input, output):
            self.activation[f'{layer}-{index}'] = output[:,index]
        handler = self.model.transformer.bert.encoder.layer[layer].output.register_forward_hook(hook)
        return handler

    def _coordinate_to_setter(self, coord):
        layer, index = coord
        def hook(model, input, output):
            # NOTE: This might lead to errors about inplace manipulations during the backprop.
            output[:,index] = self.activation[f'{layer}-{index}']
        handler = self.model.transformer.bert.encoder.layer[layer].output.register_forward_hook(hook)
        return handler

    def forward(self, source, base, coord):
        # NOTE: other ways that do not require constantly adding / removing hooks should exist
        assert source.shape == base.shape

        # set hook to get activation
        get_handler = self._coordinate_to_getter(coord)

        # get output on source examples (and also capture the activations)
        # with torch.no_grad():
        #     source_logits = self.model(source)
        source_logits_T1, source_logits_T2 = self.model(source)

        # remove the handler (don't store activations on base)
        get_handler.remove()

        # get base logits
        base_logits_T1, base_logits_T2 = self.model(base)

        # set hook to do the intervention
        set_handler = self._coordinate_to_getter(coord)

        # get counterfactual output on base examples
        counterfactual_logits_T1, counterfactual_logits_T2 = self.model(base)

        # remove the handler
        set_handler.remove()

        return source_logits_T1, base_logits_T1, counterfactual_logits_T1, source_logits_T2, base_logits_T2, counterfactual_logits_T2
