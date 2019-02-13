# Backbones

The following models have been taken from the repository https://github.com/kuangliu/pytorch-cifar: 

- densenet.py - DenseNet. This model uses dense connections where each layer receives feature maps from all preceding layers, leading to improved feature reuse and reduced vanishing gradients.
- efficientnet.py - EfficientNet. It's a family of CNNs that uses a compound scaling method to uniformly scale network depth, width, and resolution for better performance and efficiency.
- mobilenet.py - MobileNet. It's designed for mobile and embedded vision applications and uses depthwise separable convolutions to reduce computational cost.
- pnasnet.py - PNASNet. PNAS stands for Progressive Neural Architecture Search, a method used to automatically design the network's architecture.
- resnet.py - ResNet. ResNet stands for Residual Network. It uses skip connections (or residual connections) to allow gradients to flow directly through the network, which helps mitigate the vanishing gradient problem in very deep networks.
- senet.py - SENet. SE stands for Squeeze-and-Excitation. It's an attention mechanism that adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels.
- shufflenetv2.py - ShuffleNetV2. It's an efficient architecture designed with a focus on real-world inference speed by minimizing memory access cost (MAC).
- dla.py - DLA (Deep Layer Aggregation). It's a network architecture designed to merge features from different layers and resolutions, enhancing the representation of both low-level and high-level features.
- dla_simple.py - DLA (Deep Layer Aggregation). This is likely a simplified or alternative implementation of the DLA architecture.
- dpn.py - DPN (Dual Path Networks). It combines the benefits of both DenseNet and ResNet by having dual paths: one for feature reuse (like DenseNet) and another for feature exploration (like ResNet).
- googlenet.py - GoogLeNet: The description "Inception model, uses convolutions with different kernel sizes" is correct but a bit simplified. GoogLeNet is the name of the network, and it introduced the Inception module. The Inception module uses parallel convolutional filters of different sizes (1x1, 3x3, 5x5) and a pooling layer to capture multiscale features.
- lenet.py - LeNet. It's one of the earliest convolutional neural networks, primarily used for handwritten digit recognition.
- mobilenetv2.py - MobileNetV2. This is an optimized version of MobileNet that introduces the inverted residual block with linear bottlenecks to improve efficiency and performance.
- preact_resnet.py - Pre-activated ResNet. This variant of ResNet places the activation function (like ReLU) and batch normalization layers before the convolution layers, which helps improve the flow of information and training stability.
- regnet.py - RegNet. It's a family of CNNs designed to explore the design space of network architectures using a simple, quantized design space called RegNet that's optimized for efficiency and scalability.
- resnext.py - ResNeXt. It's a variant of ResNet that introduces a new dimension called "cardinality," which represents the number of parallel paths in a block, effectively acting as a form of group convolution.
- shufflenet.py - ShuffleNet. This lightweight CNN uses a channel shuffle operation to improve information flow between different groups of channels, making it efficient for low-cost devices.
- vgg.py - VGG. VGG stands for Visual Geometry Group. It's a very deep CNN model characterized by its use of small 3x3 convolutional kernels stacked on top of each other, which showed that depth is a key component for good performance.


## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |
