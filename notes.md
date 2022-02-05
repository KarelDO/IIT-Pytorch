Implementing IIT on a simple arithmetic problem:

Debugging inplace gradient modification error:
    - cloning does not help.
    - detaching does not help -> this worries me. At this point, the variable should have no computational graph associated with it, so it should definitely work. This leads me to believe the error is with the way the variable is inserted?
    - forward hooks can return a value, which will serve as the new modified output. This is better than the previous inplace operation.
    - also solved for transformers?

Requirements:
    - A task            --> Simple Arithmetic first
    - A neural model
    - A causal model    --> CHECK
        - Can this be implemented with PyTorch? So that we can easily re-use the hook code. The code requires layers on which to hook. How does this concept translate to causal models?
    - Training loop
        - Sample source and base inputs
        - train neural model on source and base
        - do IIT training
    - Evaluation loop
        - standard behavioral evaluation
        - interchange intervention accuracy
    - Datasplit that shows the benefit of IIT

Steps:
    - refactor intervention code in separate file CHECK
    - define dataloader CHECK
    - do a normal training setup without IIT CHECK
    - do training with IIT CHECK
    - implement IIAccuracy?
    - add model saving
    - add training curves (wandb?)

Remarks:
    - need to refactor Interventionable for easier use
        - task specific and counterfactual in one go?
        - make sure you can call it in the normal way?
        - eval and train functions should map to underlying neural model?


TODO:
    - fix up double forward pass in train
    - use batches in eval

Questions:
    - Is IIT slowing down training?
    - Do we need a label for the source input? Yes to asses if the intervention is impactful.