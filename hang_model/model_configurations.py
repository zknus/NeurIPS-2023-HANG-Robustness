# from function_transformer_attention import ODEFuncTransformerAtt
from .function_GAT_attention import ODEFuncAtt
from .function_laplacian_diffusion import LaplacianODEFunc
# from function_gcn import GCNFunc
from .block_transformer_attention import AttODEblock
from .block_constant import ConstantODEblock
from .function_laplacian_random import LaplacianRandomODEFunc
from .block_constant_batch import ConstantODEblockbatch
from .function_transformer_attention import ODEFuncTransformerAtt
from .function_GAT_norm import ODEFuncAttNorm
from .block_constant_time import ConstantODEblockTime
from .function_hamgcn_van import HAMGCNFunc_VAN
from .function_hamgcn_quad import HAMGCNFunc_QUAD
from .function_laplacian_grand import LaplacianODEFuncGRAND
from .function_beltrami_trans import ODEFuncBeltramiAtt
from .function_transformer_grand import ODEFuncTransformerAtt_GRAND
from .block_constant_plot import ConstantODEblock_PLOT
from .function_laplacian_grand_plot import LaplacianODEFuncGRAND_PLOT
from .function_transformer_grand_plot import ODEFuncTransformerAtt_GRAND_PLOT
from .block_constant_energy import ConstantODEblock_ENERGY
class BlockNotDefined(Exception):
    pass


class FunctionNotDefined(Exception):
    pass


def set_block(opt):
    ode_str = opt['block']
    if ode_str == 'attention':
        block = AttODEblock
    elif ode_str == 'constant':
        block = ConstantODEblock
    elif ode_str == 'constantbatch':
        block = ConstantODEblockbatch
    elif ode_str == 'constanttime':
        block = ConstantODEblockTime
    elif ode_str == 'constantplot':
        block = ConstantODEblock_PLOT
    elif ode_str == 'constantenergy':
        block = ConstantODEblock_ENERGY
    else:
        raise BlockNotDefined
    return block


def set_function(opt):
    ode_str = opt['function']
    if ode_str == 'laplacian':
        f = LaplacianODEFunc
    elif ode_str == 'GAT':
        f = ODEFuncAtt
    elif ode_str == 'laprandom':
        f = LaplacianRandomODEFunc
    elif ode_str == 'transformer':
        f = ODEFuncTransformerAtt
    elif ode_str == 'GATnorm':
        f = ODEFuncAttNorm
    elif ode_str == 'lapgrand':
        f = LaplacianODEFuncGRAND
    elif ode_str == 'belgrand':
        f = ODEFuncBeltramiAtt
    elif ode_str == 'transgrand':
        f = ODEFuncTransformerAtt_GRAND
    elif ode_str == 'lapgrandplot':
        f = LaplacianODEFuncGRAND_PLOT
    elif ode_str == 'transgrandplot':
        f = ODEFuncTransformerAtt_GRAND_PLOT

    # elif ode_str == 'gcn':
    #     f = GCNFunc
    elif ode_str == 'hang':
        f = HAMGCNFunc_VAN
    elif ode_str == 'hangquad':
        f = HAMGCNFunc_QUAD
    else:
        raise FunctionNotDefined
    return f
