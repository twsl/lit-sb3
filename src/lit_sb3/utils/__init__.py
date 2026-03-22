"""lit_sb3.utils — consolidated utility modules."""

from lit_sb3.utils.cli import StoreDict
from lit_sb3.utils.device import get_device
from lit_sb3.utils.distribution import make_masked_proba_distribution, make_proba_distribution
from lit_sb3.utils.env import create_test_env, is_wrapped, make_atari_env, make_vec_env, unwrap_wrapper
from lit_sb3.utils.logger import Logger, configure
from lit_sb3.utils.maskable import get_action_masks, is_masking_supported
from lit_sb3.utils.math import (
    conjugate_gradient_solver,
    explained_variance,
    kl_divergence,
    quantile_huber_loss,
    safe_mean,
)
from lit_sb3.utils.model import get_hf_trained_models, get_model_path, get_saved_hyperparams, get_trained_models
from lit_sb3.utils.reflection import get_callback_list, get_class_by_name, get_wrapper_class
from lit_sb3.utils.registry import ALGOS
from lit_sb3.utils.save import (
    data_to_json,
    json_to_data,
    load_from_pkl,
    load_from_zip_file,
    open_path,
    recursive_getattr,
    recursive_setattr,
    save_to_pkl,
    save_to_zip_file,
)
from lit_sb3.utils.schedule import (
    ConstantSchedule,
    FloatSchedule,
    LinearSchedule,
    SimpleLinearSchedule,
    constant_fn,
    get_linear_fn,
    get_schedule_fn,
    linear_schedule,
)
from lit_sb3.utils.sequence import create_sequencers, pad, pad_and_flatten
from lit_sb3.utils.spaces import (
    check_for_correct_spaces,
    check_shape_equal,
    is_vectorized_box_observation,
    is_vectorized_dict_observation,
    is_vectorized_discrete_observation,
    is_vectorized_multibinary_observation,
    is_vectorized_multidiscrete_observation,
    is_vectorized_observation,
)
from lit_sb3.utils.system import configure_logger, get_latest_run_id, get_system_info
from lit_sb3.utils.tensors import flat_grad, get_parameters_by_name, obs_as_tensor, polyak_update, zip_strict
from lit_sb3.utils.training import set_random_seed, should_collect_more_steps, update_learning_rate
