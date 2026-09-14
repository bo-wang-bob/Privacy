"""Audit norm diagnostics according to the recorded synthesis version."""
import math


def norm_filter_enabled(summary):
    current = summary.get('implementation') == 'local_token_geometry_v11_no_norm_filter'
    if current:
        if (summary.get('norm_ratio_filter_enabled') is not False or
                summary.get('norm_ratio_role') != 'diagnostic_only' or
                summary['options'].get('replacement_policy') != 'all' or
                {'norm_ratio_min','norm_ratio_max'} & set(summary['options'])):
            raise ValueError('Inconsistent no-norm-filter protocol metadata.')
    elif summary.get('norm_ratio_filter_enabled') is False:
        raise ValueError('Historical norm-filter protocol cannot disable its recorded constraint.')
    return not current


def valid_norm_diagnostic(ratio, summary):
    constrained = norm_filter_enabled(summary)
    if not math.isfinite(ratio) or ratio < 0:
        return False
    return not constrained or summary['options']['norm_ratio_min'] <= ratio <= summary['options']['norm_ratio_max']
