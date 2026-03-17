from gymnasium import spaces
import torch
from torch import nn

from lit_sb3.common.preprocessing import get_flattened_obs_dim, is_image_space
from lit_sb3.common.types import TensorDict
from lit_sb3.modules.feature_extractors.base import BaseFeaturesExtractor
from lit_sb3.modules.feature_extractors.nature_cnn import NatureCNN


class CombinedExtractor(BaseFeaturesExtractor):
    """Combined features extractor for Dict observation spaces.

    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    Args:
        observation_space:
        cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
            256 to avoid exploding network sizes.
        normalized_image: Whether to assume that the image is already normalized
            or not (this disables dtype and bounds checks): when True, it only checks that
            the space is a Box and has 3 dimensions.
            Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)
