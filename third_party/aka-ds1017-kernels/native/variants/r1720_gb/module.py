"""Full shifted mHC with production BF16 and FP8 outputs and both scale layouts."""
import torch
from .source_build import build

def run(sf_layout, shared_sf_block_m, **params):
    dg = build()
    x = params['x']
    m, hidden = x.shape
    outputs = dict(new_residual=torch.empty_like(params['residual']),
        new_prev_mix=torch.empty_like(params['shifted_prev_mix']),
        new_post_mix=torch.empty_like(params['post_mix']),
        new_comb_res_mix=torch.empty_like(params['comb_res_mix']),
        y_bf16=torch.empty_like(x),
        y_fp8=torch.empty_like(x, dtype=torch.float8_e4m3fn))
    if sf_layout == 'col':
        outputs['y_gemm_sf'] = torch.empty_strided((m, hidden // 128), (1, (m + 3) // 4 * 4),
                                                  dtype=torch.int32, device=x.device)
    else:
        outputs['y_routed_sf'] = torch.empty((m, hidden // 128), dtype=torch.int32, device=x.device)
        rows = (m + shared_sf_block_m - 1) // shared_sf_block_m * ((shared_sf_block_m + 127) // 128 * 128)
        storage = torch.empty_strided((rows, hidden // 128), (1, rows), dtype=torch.int32, device=x.device)
        outputs['y_shared_sf'] = storage[:m]
    dg.mega_mhc(**params, **outputs, shared_sf_block_m=shared_sf_block_m if sf_layout == 'extra' else 0)
    if sf_layout == 'extra':
        outputs['y_shared_sf_storage'] = storage
    return outputs
