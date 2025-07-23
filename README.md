# Cross-modality Material Embedding Loss for Transferring Knowledge Between Heterogeneous Material Descriptors

## Abstract
Despite the remarkable successes of transfer learning in materials science, the practicality of existing transfer learning methods are still limited in real-world applications of materials science because they essentially assume the same material descriptors on source and target materials datasets. In other words, existing transfer learning methods cannot utilize the knowledge extracted from calculated crystal structures when analyzing experimental observations of real-world chemical experiments. We propose a transfer learning criterion, called *cross-modality material embedding loss* (CroMEL), to build a source feature extractor that can transfer knowledge extracted from calculated crystal structures to prediction models in target applications where only simple chemical compositions are accessible. The prediction models based on transfer learning with CroMEL showed state-of-the-art prediction accuracy on 14 experimental materials datasets in various chemical applications. In particular, the prediction models with CroMEL achieved R2-scores greater than 0.95 in predicting the experimentally measured formation enthalpies and band gaps of the experimentally synthesized materials.

## Run
> [!NOTE]
> The source cluation dataset in ``dataset/src_calc/mps`` is an example for you implementation. Please use your source calculation datasets or download a full database of Materials Project.

- ``build_src_model.py``: A script to build a CroMEL-based source model on source calculation datasets.
- ``exec_tl.py``: A script to train and evaluate a target prediction model with the cross-modality transfer learning.
