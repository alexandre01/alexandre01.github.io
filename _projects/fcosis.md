---
layout: project
title: FCOSIS
author: Alexandre Carlier
image: fcosis_concept.png
report: FCOSIS.pdf
show_image: false
priority: 500
---

# FCOSIS: Fully-Convolutional One-Stage Instance Segmentation

We present FCOSIS, a fully convolutional single-stage anchor-free framework for instance segmentation. Our model predicts a class-agnostic, high-resolution segmentation map, similar to semantic segmentation, while preserving the benefits of multi-scale pyramidal networks for a high object detection performance. Instances are grouped in a per-pixel fashion using both the bounding box regression and a dense geometrical based embedding. Our design is conceptually simple, single-stage, and combines the benefits of object detection and semantic segmentation.