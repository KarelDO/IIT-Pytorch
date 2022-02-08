Implementing IIT on a simple arithmetic problem:


Requirements:
    - be able to show the benefit of IIT using a domain split
    - run experiments using two alignments / two causal models
        - one alignments for training
        - one alignment for evaluating / finetuning

Steps:
    - introduce a difficult domain split and show the benefit of IIT
        - decide on the domain split
        - specify it during dataset creation
    - be able to toggle IIT training using the conf file
    - add a second model and second alignment


Remarks:
    - need to refactor Interventionable for easier use
        - make sure you can call it in the normal way?
        - eval and train functions should map to underlying neural model?

    - would be cool to add an alignment specification to the ONNX file format.
        - plot model onnx to svg and add the alignment arrows? Would be really nice in the dashboard
        - annotate II accuracy on this svg with colors



Questions:
