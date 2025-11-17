
# ==================== BASELINE ====================
BASELINE = {
    'name': 'baseline',
    'type': 'standard',
    'conv1': {'in': 3, 'out': 8, 'kernel': 3, 'padding': 1},
    'conv2': {'in': 8, 'out': 12, 'kernel': 3, 'padding': 1},
    'conv3': {'in': 12, 'out': 16, 'kernel': 3, 'padding': 1},
    'conv4': {'in': 16, 'out': 12, 'kernel': 3, 'padding': 1},
    'conv5': {'in': 12, 'out': 3, 'kernel': 3, 'padding': 1},
}

# ==================== LATENT SPACE SIZE ====================

EXP_1A_SMALL_LATENT = {
    'name': 'exp_1a_small_latent',
    'type': 'standard',
    'conv1': {'in': 3, 'out': 8, 'kernel': 3, 'padding': 1},
    'conv2': {'in': 8, 'out': 12, 'kernel': 3, 'padding': 1},
    'conv3': {'in': 12, 'out': 8, 'kernel': 3, 'padding': 1},  # 16 -> 8
    'conv4': {'in': 8, 'out': 12, 'kernel': 3, 'padding': 1},   # 16 -> 8
    'conv5': {'in': 12, 'out': 3, 'kernel': 3, 'padding': 1},
}

EXP_1B_LARGE_LATENT = {
    'name': 'exp_1b_large_latent',
    'type': 'standard',
    'conv1': {'in': 3, 'out': 8, 'kernel': 3, 'padding': 1},
    'conv2': {'in': 8, 'out': 12, 'kernel': 3, 'padding': 1},
    'conv3': {'in': 12, 'out': 32, 'kernel': 3, 'padding': 1},  # 16 -> 32
    'conv4': {'in': 32, 'out': 12, 'kernel': 3, 'padding': 1},  # 16 -> 32
    'conv5': {'in': 12, 'out': 3, 'kernel': 3, 'padding': 1},
}

EXP_1C_TINY_LATENT = {
    'name': 'exp_1c_tiny_latent',
    'type': 'standard',
    'conv1': {'in': 3, 'out': 8, 'kernel': 3, 'padding': 1},
    'conv2': {'in': 8, 'out': 12, 'kernel': 3, 'padding': 1},
    'conv3': {'in': 12, 'out': 4, 'kernel': 3, 'padding': 1},   # 16 -> 4
    'conv4': {'in': 4, 'out': 12, 'kernel': 3, 'padding': 1},   # 16 -> 4
    'conv5': {'in': 12, 'out': 3, 'kernel': 3, 'padding': 1},
}

# ==================== DEPTH ====================

EXP_2A_SHALLOW = {
    'name': 'exp_2a_shallow',
    'type': 'shallow',
    'conv1': {'in': 3, 'out': 12, 'kernel': 3, 'padding': 1},
    'conv2': {'in': 12, 'out': 16, 'kernel': 3, 'padding': 1},
    'conv3': {'in': 16, 'out': 3, 'kernel': 3, 'padding': 1},
}

EXP_2B_DEEPER = {
    'name': 'exp_2b_deeper',
    'type': 'deep',
    'conv1': {'in': 3, 'out': 8, 'kernel': 3, 'padding': 1},
    'conv2': {'in': 8, 'out': 12, 'kernel': 3, 'padding': 1},
    'conv3': {'in': 12, 'out': 16, 'kernel': 3, 'padding': 1},
    'conv4': {'in': 16, 'out': 20, 'kernel': 3, 'padding': 1},
    'conv5': {'in': 20, 'out': 16, 'kernel': 3, 'padding': 1},
    'conv6': {'in': 16, 'out': 12, 'kernel': 3, 'padding': 1},
    'conv7': {'in': 12, 'out': 3, 'kernel': 3, 'padding': 1},
}

# ==================== WIDTH ====================

EXP_3A_NARROW = {
    'name': 'exp_3a_narrow',
    'type': 'standard',
    'conv1': {'in': 3, 'out': 4, 'kernel': 3, 'padding': 1},
    'conv2': {'in': 4, 'out': 6, 'kernel': 3, 'padding': 1},
    'conv3': {'in': 6, 'out': 8, 'kernel': 3, 'padding': 1},
    'conv4': {'in': 8, 'out': 6, 'kernel': 3, 'padding': 1},
    'conv5': {'in': 6, 'out': 3, 'kernel': 3, 'padding': 1},
}

EXP_3B_WIDE = {
    'name': 'exp_3b_wide',
    'type': 'standard',
    'conv1': {'in': 3, 'out': 16, 'kernel': 3, 'padding': 1},
    'conv2': {'in': 16, 'out': 24, 'kernel': 3, 'padding': 1},
    'conv3': {'in': 24, 'out': 32, 'kernel': 3, 'padding': 1},
    'conv4': {'in': 32, 'out': 24, 'kernel': 3, 'padding': 1},
    'conv5': {'in': 24, 'out': 3, 'kernel': 3, 'padding': 1},
}

# ==================== KERNEL SIZE ====================

EXP_4A_LARGE_KERNELS = {
    'name': 'exp_4a_large_kernels',
    'type': 'standard',
    'conv1': {'in': 3, 'out': 8, 'kernel': 5, 'padding': 2},
    'conv2': {'in': 8, 'out': 12, 'kernel': 5, 'padding': 2},
    'conv3': {'in': 12, 'out': 16, 'kernel': 5, 'padding': 2},
    'conv4': {'in': 16, 'out': 12, 'kernel': 5, 'padding': 2},
    'conv5': {'in': 12, 'out': 3, 'kernel': 5, 'padding': 2},
}

ALL_EXPERIMENTS = [
    BASELINE,
    EXP_1A_SMALL_LATENT,
    EXP_1B_LARGE_LATENT,
    EXP_1C_TINY_LATENT,
    EXP_2A_SHALLOW,
    EXP_2B_DEEPER,
    EXP_3A_NARROW,
    EXP_3B_WIDE,
    EXP_4A_LARGE_KERNELS,
]