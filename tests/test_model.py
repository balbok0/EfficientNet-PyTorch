from collections import OrderedDict

import pytest
import torch
from torch import nn

from efficientnet_pytorch import EfficientNet

# -- fixtures -------------------------------------------------------------------------------------


@pytest.fixture(scope='module', params=list(range(4)))
def model(request):
    return f'efficientnet-b{request.param}'


@pytest.fixture(scope='module', params=[True, False])
def pretrained(request):
    return request.param


@pytest.fixture(scope='function')
def net(model, pretrained):
    return EfficientNet.from_pretrained(model) if pretrained else EfficientNet.from_name(model)


# -- tests ----------------------------------------------------------------------------------------


@pytest.mark.parametrize('img_size', [224, 256, 512])
def test_forward(net, img_size):
    """Test `.forward()` doesn't throw an error."""
    data = torch.zeros((1, 3, img_size, img_size))
    output = net(data)
    assert not torch.isnan(output).any()


def test_dropout_training(net):
    """Test dropout `.training` is set by `.train()` on parent `nn.module`"""
    net.train()
    assert net._dropout.training is True


def test_dropout_eval(net):
    """Test dropout `.training` is set by `.eval()` on parent `nn.module`"""
    net.eval()
    assert net._dropout.training is False


def test_dropout_update(net):
    """Test dropout `.training` is updated by `.train()` and `.eval()` on parent `nn.module`"""
    net.train()
    assert net._dropout.training is True
    net.eval()
    assert net._dropout.training is False
    net.train()
    assert net._dropout.training is True
    net.eval()
    assert net._dropout.training is False


@pytest.mark.parametrize('img_size', [224, 256, 512])
def test_modify_dropout(net, img_size):
    """Test ability to modify dropout and fc modules of network."""
    dropout = nn.Sequential(
        OrderedDict(
            [
                ('_bn2', nn.BatchNorm1d(net._bn1.num_features)),
                ('_drop1', nn.Dropout(p=net._global_params.dropout_rate)),
                ('_linear1', nn.Linear(net._bn1.num_features, 512)),
                ('_relu', nn.ReLU()),
                ('_bn3', nn.BatchNorm1d(512)),
                ('_drop2', nn.Dropout(p=net._global_params.dropout_rate / 2)),
            ]
        )
    )
    fc = nn.Linear(512, net._global_params.num_classes)

    net._dropout = dropout
    net._fc = fc

    data = torch.zeros((2, 3, img_size, img_size))
    output = net(data)
    assert not torch.isnan(output).any()


@pytest.mark.parametrize('img_size', [224, 256, 512])
def test_modify_pool(net, img_size):
    """Test ability to modify pooling module of network."""

    class AdaptiveMaxAvgPool(nn.Module):
        def __init__(self):
            super().__init__()
            self.ada_avgpool = nn.AdaptiveAvgPool2d(1)
            self.ada_maxpool = nn.AdaptiveMaxPool2d(1)

        def forward(self, x):
            avg_x = self.ada_avgpool(x)
            max_x = self.ada_maxpool(x)
            x = torch.cat((avg_x, max_x), dim=1)
            return x

    avg_pooling = AdaptiveMaxAvgPool()
    fc = nn.Linear(net._fc.in_features * 2, net._global_params.num_classes)

    net._avg_pooling = avg_pooling
    net._fc = fc

    data = torch.zeros((2, 3, img_size, img_size))
    output = net(data)
    assert not torch.isnan(output).any()


@pytest.mark.parametrize('img_size', [224, 256, 512])
def test_extract_endpoints(net, img_size):
    """Test `.extract_endpoints()` doesn't throw an error."""
    data = torch.zeros((1, 3, img_size, img_size))
    endpoints = net.extract_endpoints(data)
    assert not torch.isnan(endpoints['reduction_1']).any()
    assert not torch.isnan(endpoints['reduction_2']).any()
    assert not torch.isnan(endpoints['reduction_3']).any()
    assert not torch.isnan(endpoints['reduction_4']).any()
    assert not torch.isnan(endpoints['reduction_5']).any()
    assert endpoints['reduction_1'].size(2) == img_size // 2
    assert endpoints['reduction_2'].size(2) == img_size // 4
    assert endpoints['reduction_3'].size(2) == img_size // 8
    assert endpoints['reduction_4'].size(2) == img_size // 16
    assert endpoints['reduction_5'].size(2) == img_size // 32


@pytest.mark.parametrize('img_size', [224, 256, 512])
def test_script(net, img_size):
    data = torch.zeros((1, 3, img_size, img_size))
    scripted_net = torch.jit.script(net)
    assert isinstance(scripted_net, torch.jit.ScriptModule)
    # Efficient net is non-deterministic sometimes. Ignore correctness. Mae sure it runs
    # net_result = net(data.clone())
    result = scripted_net(data.clone())
    endpoints = scripted_net.extract_endpoints(data.clone())

    assert not torch.isnan(result).any()
    assert not torch.isnan(endpoints['reduction_1']).any()
    assert not torch.isnan(endpoints['reduction_2']).any()
    assert not torch.isnan(endpoints['reduction_3']).any()
    assert not torch.isnan(endpoints['reduction_4']).any()
    assert not torch.isnan(endpoints['reduction_5']).any()
    # torch.testing.assert_close(net_result, scripted_result)


@pytest.mark.parametrize('img_size', [224, 256, 512])
def test_script_freeze_optimize(net, img_size):
    data = torch.zeros((1, 3, img_size, img_size))

    scripted_net = torch.jit.script(net)
    freeze_net = torch.jit.freeze(scripted_net.eval(), preserved_attrs=["extract_endpoints"])
    optimized_net = torch.jit.optimize_for_inference(freeze_net)

    assert isinstance(optimized_net, torch.jit.ScriptModule)
    result = optimized_net(data.clone())
    endpoints = optimized_net.extract_endpoints(data.clone())
    assert not torch.isnan(result).any()
    assert not torch.isnan(endpoints['reduction_1']).any()
    assert not torch.isnan(endpoints['reduction_2']).any()
    assert not torch.isnan(endpoints['reduction_3']).any()
    assert not torch.isnan(endpoints['reduction_4']).any()
    assert not torch.isnan(endpoints['reduction_5']).any()
    # Efficient net is non-deterministic sometimes. Ignore
    # net_result = net(data.clone())
    # scripted_result = scripted_net(data.clone())
    # torch.testing.assert_close(net_result, scripted_result)
