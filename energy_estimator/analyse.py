import torch
import fairseq

bitwidth_to_minvalue = {
    32: 2**-126,
    16: 2**-30,
    8: 2**-14,
}


class HardwareModel:
    """
    Default energy figures taken from here (45nm):
    https://ieeexplore.ieee.org/document/6757323

    Assuming all operations are 32 bit floating point (others are available).
    """

    def __init__(self, optim=True):
        # Cost of a single memory access in pJ.
        # ASSUMPTION: DRAM costs dominate (on-chip caches are negligible).
        # ASSUMPTION: reads and writes cost the same.
        self.memory_cost = 1950.0

        # Cost of a single computation (e.g. multiply) in pJ.
        # ASSUMPTION: all computations cost the same amount.
        self.compute_cost = 3.7

        # Is the hardware able to optimise memory access for sparse data?
        # ASSUMPTION: there is no overhead to (de)compression.
        self.compress_sparse_weights = optim
        self.compress_sparse_activations = optim

        # Is the hardware able to skip computations when one input is zero?
        # ASSUMPTION: zeros are uniformly distributed throughout the data.
        self.compute_skip_zero_weights = optim
        self.compute_skip_zero_activations = optim


def analyse(model, input_data, hardware=HardwareModel()):
    """
    Estimate how much energy will be consumed by applying a given model to the
    given data.

    :param model: a torch.nn.Module to be analysed.
    :param input_data: a torch.Tensor to be passed through the model.
    :param hardware: a HardwareModel representing the processor on which the
                     model will be executed.
    :return: the estimated number of picojoules consumed.
    """
    stats = StatsRecorder()

    # Prepare the model for analysis.
    hooks = add_hooks(model, stats)

    # Send the data through the model.
    _ = model(input_data)

    # Clean up.
    remove_hooks(hooks)

    energy = get_energy_estimate(stats, hardware)
    print(energy, "pJ")
    return energy


def add_hooks(model, stats):
    """
    Prepare a model for analysis.

    Intercept computation in each leaf node of the network, and collect data
    on the amount of data accessed and computation performed.

    ASSUMPTION: nothing significant happens in modules which contain other
    modules. Only leaf modules are analysed.

    :param model: a torch.nn.Module to be analysed.
    :param stats: a StatsRecorder into which the results will be stored.
    """
    hooks = []

    leaf_nodes = [module for module in model.modules()
                  if len(list(module.children())) == 0]

    stat_fn = record_stats(stats)
    for module in leaf_nodes:
        hook = module.register_forward_hook(stat_fn)
        hooks.append(hook)

    return hooks


def remove_hooks(hooks):
    """
    Remove hooks from a model.

    :param hooks: an Iterable containing hooks to be removed.
    """
    for hook in hooks:
        hook.remove()


class StatsRecorder:
    def __init__(self, bitwidth=32):
        self.total_input_activations = 0.0
        self.non_zero_input_activations = 0.0
        self.total_output_activations = 0.0
        self.non_zero_output_activations = 0.0
        self.total_parameters = 0.0
        self.non_zero_parameters = 0.0
        self.computations = 0.0

        if bitwidth not in bitwidth_to_minvalue:
            raise "Passed bitwidth is not supported"
        self.min_value = bitwidth_to_minvalue[bitwidth]
        self.nonzero_func = lambda x:\
            float(len((x.abs() > self.min_value).nonzero()))

    def __reset__(self):
        del self.total_input_activations, self.non_zero_input_activations
        del self.total_output_activations, self.non_zero_output_activations
        del self.total_parameters, self.non_zero_parameters, self.computations
        self.__init__()

def get_energy_estimate(stats, hw):
    """
    Estimate the energy consumption in picojoules of a given computation on
    given hardware.

    ASSUMPTIONS:
    * Weights are read from DRAM exactly once.
    * Input activations are read from DRAM exactly once.
    * Output activations are written to DRAM exactly once.

    :param stats: a StatsRecorder containing details of the computation.
    :param hw: a HardwareModel containing details of the processor.
    """
    total = 0.0

    if hw.compress_sparse_weights:
        total += hw.memory_cost * stats.non_zero_parameters
    else:
        total += hw.memory_cost * stats.total_parameters

    if hw.compress_sparse_activations:
        total += hw.memory_cost * (stats.non_zero_input_activations +
                                   stats.non_zero_output_activations)
    else:
        total += hw.memory_cost * (stats.total_input_activations +
                                   stats.total_output_activations)

    compute_fraction = 1.0

    if hw.compute_skip_zero_weights:
        compute_fraction *= (stats.non_zero_parameters / stats.total_parameters)

    if hw.compute_skip_zero_activations:
        compute_fraction *= (stats.non_zero_input_activations /
                             stats.total_input_activations)

    total += compute_fraction * stats.computations * hw.compute_cost

    return total

def record_stats(stats):
    """
    Create a forward hook function which will record information about a layer's
    execution.

    For all module parameters/buffers, in_data and out_data, record:
    * Number of values
    * Number of non-zeros
    Also estimate amount of computation (depends on layer type).

    :param stats: a StatsRecorder to store results in.
    :return: forward hook function.
    """

    def hook_fn(nonzero_func, module, in_data, out_data):
        # Activations are sometimes Tensors, and sometimes tuples of Tensors.
        # Ensure we're always dealing with tuples.
        if isinstance(in_data, torch.Tensor):
            in_data = (in_data,)
        if isinstance(out_data, torch.Tensor):
            out_data = (out_data,)

        # Collect memory statistics.
        for tensor in in_data:
            stats.total_input_activations += tensor.numel()
            stats.non_zero_input_activations += nonzero_func(tensor)

        for tensor in out_data:
            stats.total_output_activations += tensor.numel()
            stats.non_zero_output_activations += nonzero_func(tensor)

        for tensor in module.buffers():
            stats.total_parameters += tensor.numel()
            stats.non_zero_parameters += nonzero_func(tensor)

        for tensor in module.parameters():
            stats.total_parameters += tensor.numel()
            stats.non_zero_parameters += nonzero_func(tensor)

        # Collect computation statistics.
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            # One computation per input pixel - window size is chosen adaptively
            # and windows never overlap (?).
            assert len(in_data) == 1
            input_size = in_data[0].numel()
            stats.computations += input_size

        elif isinstance(module, torch.nn.Embedding):
            stats.total_parameters += module.embedding_dim * in_data[0].numel()
            stats.non_zero_parameters += nonzero_func(out_data[0])
        elif isinstance(module, fairseq.modules.sinusoidal_positional_embedding.SinusoidalPositionalEmbedding):
            #stats.total_parameters += module.embedding_dim * in_data[0].numel()
            #stats.non_zero_parameters += len(out_data[0].nonzero())
            pass
        elif isinstance(module, torch.nn.AvgPool2d) or \
                isinstance(module, torch.nn.MaxPool2d):
            # Each output pixel requires computations on a 2D window of input.
            if type(module.kernel_size) == int:
                # Kernel size here can be either a single int for square kernel
                # or a tuple (see
                # https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d )
                window_size = module.kernel_size**2
            else:
                window_size = module.kernel_size[0] * module.kernel_size[1]

            # Not sure which output tensor to use if there are multiple of them.
            assert len(out_data) == 1
            output_size = out_data[0].numel()
            stats.computations += output_size * window_size

        elif isinstance(module, torch.nn.Conv2d):
            # Each output pixel requires computations on a 3D window of input.
            # Not sure which input tensor to use if there are multiple of them.
            assert len(in_data) == 1
            _, channels, _, _ = in_data[0].size()
            window_size = \
                module.kernel_size[0] * module.kernel_size[1] * channels

            # Not sure which output tensor to use if there are multiple of them.
            assert len(out_data) == 1
            output_size = out_data[0].numel()

            stats.computations += output_size * window_size

        elif isinstance(module, torch.nn.Dropout2d) or isinstance(module,
                torch.nn.modules.dropout.Dropout):
            # Do nothing - dropout has no effect during inference.
            pass

        elif isinstance(module, torch.nn.Linear):
            # One computation per weight, for each batch element.

            # Not sure which input tensor to use if there are multiple of them.
            assert len(in_data) == 1
            batch = in_data[0].numel() / in_data[0].shape[-1]

            stats.computations += module.weight.numel() * batch

        elif isinstance(module, torch.nn.modules.activation.ReLU) or isinstance(module, torch.nn.modules.activation.ReLU6):
            # ReLU does a single negation check
            pass

        elif isinstance(module, torch.nn.LayerNorm):
            # You first compute
            pass

        elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):

            # Accesses to E[x] and Var[x] (all channel size)

            stats.total_parameters += 2 * module.num_features
            stats.non_zero_parameters +=\
              nonzero_func(module.running_mean)+\
              nonzero_func(module.running_var)

            # (x-running_mean)/running variance
            # multiply by gamma and beta addition
            stats.computations += 4*in_data[0].numel()
        else:
            print("Unsupported module type for energy analysis:", type(module))

    return lambda *x: hook_fn(stats.nonzero_func, *x)
