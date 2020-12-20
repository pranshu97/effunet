# EffUNet

A segmentation model based on UNet architecture with an efficientnet encoder. [EfficientNet](https://arxiv.org/abs/1905.11946) B0 through B7 are supported.

Special thanks to [lukemelas](https://github.com/lukemelas) for the pytorch implementation of [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)

### Quick start guide
Install with `pip install effunet`

    from effunet import EffUNet
    model = EffUNet(model='b0',out_channels=1,freeze_backbone=True,pretrained=True,device='cuda')
and you're good to go...

License
----

MIT
