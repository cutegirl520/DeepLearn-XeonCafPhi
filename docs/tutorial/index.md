---
title: Caffe Tutorial
---
# Caffe Tutorial

Caffe is a deep learning framework and this tutorial explains its philosophy, architecture, and usage.
This is a practical guide and framework introduction, so the full frontier, context, and history of deep learning cannot be covered here.
While explanations will be given where possible, a background in machine learning and neural networks is helpful.

## Philosophy

In one sip, Caffe is brewed for

- Expression: models and optimizations are defined as plaintext schemas instead of code.
- Speed: for research and industry alike speed is crucial for state-of-the-art models and massive data.
- Modularity: new tasks and settings require flexibility and extension.
- Openness: scientific and applied progress call for common code, reference models, and reproducibility.
- Community: academic research, startup prototypes, and industrial applications all share strength by joint discussion and development in a BSD-2 project.

and these principles direct the project.

## Tour

- [Nets, Layers, and Blobs](net_layer_blob.html): the anatomy of a Caffe model.
- [Forward / Backward](forward_backward.html): the essential computations of layered compositional models.
- [Loss](loss.html): the task to be learned is defined by the loss.
- [Solver](solver.html): the solver coordinates model optimization.
- [Layer Catalogue](layers.html): the layer is the fundamental unit of modeling and computation -- Caffe's catalogue includes layers for state-of-the-art models.
- [Interfaces]