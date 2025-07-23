# Cross-modality Material Embedding Loss for Transferring Knowledge Between Heterogeneous Material Descriptors

## Abstract
Despite remarkable successes of transfer learning in materials science, the practicalities of existing transfer learning methods are still limited in real-world applications of materials science because they assume the same material descriptors on source and target materials datasets. In other words, existing transfer learning methods are not able to transfer knowledge extracted from calculated crystal structures to experimentally collected materials data. This paper proposes an efficient and universally applicable optimization criterion called *cross-modality material embedding loss* (CroMEL). CroMEL provides an objective function to build a source feature extractor that can transfer knowledge extracted from calculated crystal structures to prediction models of target applications where only simple material descriptors for experimentally synthesized materials are accessible. The prediction models generated with CroMEL showed state-of-the-art prediction accuracy on 14 materials datasets collected experimentally in various chemical applications. In particular, the prediction models generated with CroMEL achieved $R^2$-scores greater than 0.95 in predicting the experimentally measured formation enthalpies and band gaps from the chemical compositions of the materials. All source codes of this work are publicly available at https://github.com/ngs00/CroMEL.

## Run
- ``exec_tl.py``: A script to train and evaluate the target prediction models with the cross-modality transfer learning.

> [!WARNING]
> Critical content comes here.
