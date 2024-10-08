import torch
import torch.nn as nn
from typing import List, Tuple

class SharedMLP(nn.Sequential):
    def __init__(
            self,
            args: List[int],  # Feature Progation模块的需要用到的；这个就是用于创建网络时各层的维度大小，用一个列表存下各层的维度
            *,
            bn: bool = False,  # 传过来的参数是True
            activation=nn.ReLU(inplace=True),  # inplace = True ,会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
            preact: bool = False,
            first: bool = False,
            name: str = ""
    ):
        super().__init__()

        for i in range(len(args) - 1):  # 除了最后一个Feature Progation模块是4外，前面三个模块都是3
            self.add_module(            # self.a
                name + 'layer{}'.format(i),  # "layer0", "layer1"
                # 这创建的是一个全连接层
                Conv2d(
                    args[i],            # 输入的维度
                    args[i + 1],        # 输出的维度
                    bn=(not first or not preact or (i != 0)) and bn,  
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact
                )
            )


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class BatchNorm3d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


class _ConvBase(nn.Sequential):

    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name=""
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)


class Conv1d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name
        )


class Conv2d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name
        )


class Conv3d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int, int] = (1, 1, 1),
            stride: Tuple[int, int, int] = (1, 1, 1),
            padding: Tuple[int, int, int] = (0, 0, 0),
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv3d,
            batch_norm=BatchNorm3d,
            bias=bias,
            preact=preact,
            name=name
        )


class FC(nn.Sequential):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=None,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant_(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

def set_bn_momentum_default(bn_momentum):
    # 这个函数的作用是将输入的 PyTorch 模块中的 BN 层动量设置为 bn_momentum
    # 其中 m 表示输入的 PyTorch 模块，如果 m 是一个 1D、2D 或 3D 的 BN 层，则将其动量设置为 bn_momentum
    # 如果 m 是一个网络模型，则上述代码中的作用是将模型中所有的 BN 层动量设置为 bn_momentum。
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # 在 Batch Normalization 中，momentum 是控制滑动平均的超参数，用于计算每个特征通道的均值和方差的滑动平均值。
            # 具体地，momentum 表示上一次计算的滑动平均值的权重，常取一个小于 1 的数值。比如，当 momentum=0.1 时，
            # 当前时刻的均值和方差会占 90% 的权重，而上一时刻的均值和方差则占剩余的 10% 的权重。
            m.momentum = bn_momentum

    return fn

# 在 BNMomentumScheduler 类的构造函数中，将传入的参数初始化并调用了 step 方法。在 step 方法中，
# 会更新 self.last_epoch 的值，然后通过 setter 函数将当前 epoch 对应的 BN 层动量值设置到模型的所有 BN 层上。
# 这里的 setter 函数是一个以 BN 动量为参数的函数，返回一个函数，该返回函数接受一个 PyTorch 模型中的子模块作为参数，
# 并将其 BN 层的动量设置为参数中的 BN 动量。

 # 因此，BNMomentumScheduler 的作用是动态调整模型中 BN 层的动量值，从而优化模型的训练效果。
 # 其中 bn_lambda 是一个以 epoch 为参数的函数，用于计算当前 epoch 下的 BN 层动量值。
 # 在模型训练过程中，可以通过不断调用 BNMomentumScheduler 对象的 step 方法来更新 BN 层的动量，从而实现更好的训练效果。

class BNMomentumScheduler(object):
    # 刚开始创建时，last_epoch被传入的参数是start_epoch-1=0-1=-1，也就是刚开始创建的时候last_epoch就是-1
    # 只要模型训练过程中没有中断过，使得start_epoch值变化，last_epoch就是-1
    # model就是graspnet的网络模型对象net
    # bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * cfgs.bn_decay_rate**(int(it / cfgs.bn_decay_step)), BN_MOMENTUM_MAX)
    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)  # 注意后面每调用一次step()都会让self.last_epoch加1
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch  # 创建对象时接到的参数时last_epoch+1=-1+1=0
        # self.lmbd(epoch)中如果epoch为0，也就是参数it=0， 则输出值为：max(0.5 * 0.5^(int(0/2))， 0.001)=0.5
        # self.setter(0.5)将对应的模型中所有(nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)模块的动量设置为0.5
        self.model.apply(self.setter(self.lmbd(epoch)))    


