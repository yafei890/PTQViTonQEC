# PTQViTonQEC

Implementation of **Post-Training Quantization of Vision Transformers Based on Quantization Error Compensation** by Yafei Du, Zhao Li, and Yuting Yan.

## Overview

This work studies low-bit post-training quantization of vision transformers. The proposed method combines improved parameter search for adaptive logarithmic activation quantization with activation-error-aware second-order weight compensation.

The activation parameter search combines low-error candidates with randomly selected candidates to balance search coverage and local refinement. Weight reconstruction uses quantized inputs and floating-point reference outputs to account for activation-induced errors, followed by compensation for weight quantization errors using inverse-Hessian information. All compensation is performed offline and incorporated into the quantized weights, without introducing additional compensation branches during inference.

## Code Availability

The full implementation of the proposed method is publicly available in this repository:

[https://github.com/DontStudy/PTQViTonQEC](https://github.com/DontStudy/PTQViTonQEC)

The code accompanies the manuscript and is provided to support inspection of the method and reproduction of the reported experiments. Experimental results should be interpreted together with the model configurations, calibration settings, and evaluation protocols described in the manuscript.

## Acknowledgments

This implementation builds on **AdaLog** and **OPTQ**. We thank their authors for making their research and code publicly available. Our contributions concern the activation parameter search strategy and activation-error-aware second-order weight compensation described in the manuscript. The underlying methods and reused code remain attributed to their respective authors.

## Datasets

The experiments reported in the manuscript use **ImageNet-1K** for image classification and **COCO 2017** for object detection and instance segmentation. Please obtain these datasets from their official sources and follow the corresponding access and usage requirements:

- [ImageNet](https://www.image-net.org/)
- [COCO](https://cocodataset.org/)

## Citation

If you use this implementation in your research, please cite the associated manuscript and the relevant underlying work, including AdaLog and OPTQ.

**Yafei Du, Zhao Li, and Yuting Yan.** *Post-Training Quantization of Vision Transformers Based on Quantization Error Compensation.*

## Contact

For questions about the method or implementation, please open an issue in this repository or contact Zhao Li at [lizhao@sdut.edu.cn](mailto:lizhao@sdut.edu.cn).
