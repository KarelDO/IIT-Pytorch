# IIT-DRAFT
This codebase contains two notebooks `interventions` and `interventions-transformers` to illustrate how to use PyTorch to do interventions on a simple model and HuggingFace Transformer respectively.

We use PyTorch to define the computational graph of the model on which we will intervene. PyTorch allows us to attach hooks during the forward execution of the computational graph using the `register_forward_hook` method. These hooks will allow us to easily define the interventions while letting PyTorch do all the heavy lifting.

To get the value of a specific node on a source input and insert that value in the computational graph on a base input, we will use two hooks. We attach one hook to the specific node during the forward pass on the source input to save the output of that node in a dictionary. When running the model on the base input, we remove the first hook and attach a second hook that sets the output of the node equal to the value saved in the dictionary. This implements the intervention behavior.

A new class is created to abstract away the attachment of the hooks to the model. The user can now perform an intervention by supplying a batch of base and source inputs and specifying on which node to intervene.

In the `interventions` notebook, the user can specify the layer on which to intervene by using the name PyTorch associates with the layer. In `interventions-transformers`, the user needs to supply two extra functions (`_coordinate_to_getter` and `_coordinate_to_setter`) which specify the coordinate system over the internal representations. The user can now use these coordinates to specify an intervention. The code also supports intervening on rectangular slices of nodes by supplying a bottom left and top right coordinate during the intervention process.
