# IIT-DRAFT

## Notebooks
This codebase contains three notebooks `interventions`, `interventions-causalmodels`, and `interventions-transformers` to illustrate how to use PyTorch to do interventions on a simple model, causal model, and HuggingFace Transformer respectively.

We use PyTorch to define the computational graph of the model on which we will intervene. PyTorch allows us to attach hooks during the forward execution of the computational graph using the `register_forward_hook` method. These hooks will allow us to easily define the interventions while letting PyTorch do all the heavy lifting.

To get the value of a specific node on a source input and insert that value in the computational graph on a base input, we will use two hooks. We attach one hook to the specific node during the forward pass on the source input to save the output of that node in a dictionary. When running the model on the base input, we remove the first hook and attach a second hook that sets the output of the node equal to the value saved in the dictionary. This implements the intervention behavior.

A new class is created to abstract away the attachment of the hooks to the model. The user can now perform an intervention by supplying a batch of base and source inputs and specifying on which node to intervene.

In the `interventions` notebook, the user can specify the layer on which to intervene by using the name PyTorch associates with the layer. In `interventions-transformers`, the user needs to supply two extra functions (`_coordinate_to_getter` and `_coordinate_to_setter`) which specify the coordinate system over the internal representations. The user can now use these coordinates to specify an intervention. The code also supports intervening on rectangular slices of nodes by supplying a bottom left and top right coordinate during the intervention process.

Instead of implementing the above mentioned coordinate functions, the user can also make use of `torch.nn.Identity()` layers to get access to the desired intermediate variables. Hooks need to be attached to layers. Since the causal model is deterministic, it features no model layers. By adding these identity dummy layers, we create new anchorpoints for hooks. There might be a better way to implement hookable causal models in PyTorch.

### Inplace gradient modification error during backpropagation
Manipulating the computational graph using the forward hooks can cause errors during backpropagation, if the output of the forward layer was manipulated via an inplace operation. This is the case in the `interventions-transformers` notebook. I still need to test if these intervened transformers error during backpropagation.

### Remaining TODOs
- The `Interventionable` class should be refactored to be a PyTorch module itself. If this would be the case, all subsequent code could just use the typical model API. At this point, the subsequent code needs to constantly refer to the underlying PyTorch model of the `Interventionable` class.
- We are not assessing the impactfulness of the intervention yet.
- Use the ONNX format to generate an SVG of the causal and neural model where their alignment is annotated. This will make it much easier to communicate about experiments in the future.

## Arithmetic


