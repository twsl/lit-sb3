import torch
from torch import nn

from lit_sb3.utils.device import get_device


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: list[int],
    activation_fn: type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
    pre_linear_modules: list[type[nn.Module]] | None = None,
    post_linear_modules: list[type[nn.Module]] | None = None,
) -> list[nn.Module]:
    """Create a multi layer perceptron (MLP), which is.

    a collection of fully-connected layers each followed by an activation function.

    Args:
        input_dim: Dimension of the input vector
        output_dim: Dimension of the output (last layer, for instance, the number of actions)
        net_arch: Architecture of the neural net
            It represents the number of units per layer.
            The length of this list is the number of layers.
        activation_fn: The activation function
            to use after each layer.
        squash_output: Whether to squash the output using a Tanh
            activation function
        with_bias: If set to False, the layers will not learn an additive bias
        pre_linear_modules: List of nn.Module to add before the linear layers.
            These modules should maintain the input tensor dimension (e.g. BatchNorm).
            The number of input features is passed to the module's constructor.
            Compared to post_linear_modules, they are used before the output layer (output_dim > 0).
        post_linear_modules: List of nn.Module to add after the linear layers
            (and before the activation function). These modules should maintain the input
            tensor dimension (e.g. Dropout, LayerNorm). They are not used after the
            output layer (output_dim > 0). The number of input features is passed to
            the module's constructor.

    Returns:
        The list of layers of the neural network
    """
    pre_linear_modules = pre_linear_modules or []
    post_linear_modules = post_linear_modules or []

    modules = []
    if len(net_arch) > 0:
        # BatchNorm maintains input dim
        for module in pre_linear_modules:
            modules.append(module(input_dim))

        modules.append(nn.Linear(input_dim, net_arch[0], bias=with_bias))

        # LayerNorm, Dropout maintain output dim
        for module in post_linear_modules:
            modules.append(module(net_arch[0]))

        modules.append(activation_fn())

    for idx in range(len(net_arch) - 1):
        for module in pre_linear_modules:
            modules.append(module(net_arch[idx]))

        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))

        for module in post_linear_modules:
            modules.append(module(net_arch[idx + 1]))

        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        # Only add BatchNorm before output layer
        for module in pre_linear_modules:
            modules.append(module(last_layer_dim))

        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class MlpExtractor(nn.Module):
    """Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly.

    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers
    in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    Args:
        feature_dim: Dimension of the feature vector (can be the output of a CNN)
        net_arch: The specification of the policy and value networks.
            See above for details on its formatting.
        activation_fn: The activation function to use for the networks.
        device: PyTorch device.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: list[int] | dict[str, list[int]],
        activation_fn: type[nn.Module],
        device: torch.device | str = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net: list[nn.Module] = []
        value_net: list[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specified, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns.

        latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``.
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


def get_actor_critic_arch(net_arch: list[int] | dict[str, list[int]]) -> tuple[list[int], list[int]]:
    """Get the actor and critic network architectures for off-policy actor-critic algorithms (SAC, TD3, DDPG).

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers,
    which can be different for the actor and the critic.
    It is assumed to be a list of ints or a dict.

    1. If it is a list, actor and critic networks will have the same architecture.
        The architecture is represented by a list of integers (of arbitrary length (zero allowed))
        each specifying the number of units per layer.
       If the number of ints is zero, the network will be linear.
    2. If it is a dict,  it should have the following structure:
       ``dict(qf=[<critic network architecture>], pi=[<actor network architecture>])``.
       where the network architecture is a list as described in 1.

    For example, to have actor and critic that share the same network architecture,
    you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).

    If you want a different architecture for the actor and the critic,
    then you can specify ``net_arch=dict(qf=[400, 300], pi=[64, 64])``.

    .. note::
        Compared to their on-policy counterparts, no shared layers (other than the features extractor)
        between the actor and the critic are allowed (to prevent issues with target networks).

    Args:
        net_arch: The specification of the actor and critic networks.
            See above for details on its formatting.

    Returns:
        The network architectures for the actor and the critic
    """
    if isinstance(net_arch, list):
        actor_arch, critic_arch = net_arch, net_arch
    else:
        assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
    return actor_arch, critic_arch


__all__ = ["create_mlp", "get_actor_critic_arch", "MlpExtractor"]
