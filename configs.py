common = {
    "pitch_x" : 150.0, # microns
    "pitch_y" : 100.0, # microns
    "batch_size" : 256,
    "epochs" : 5000,
    "loss_name": "nll_loss",
    "input_file": "test_clusters.root",
    "debug_predictions": False,
    "modelname": "conv_model",
    "output_scale": 50.,
    "first_guess": "center", # 'center' or 'generic'
    }

layer_configs = {}
for layer in ["L1U","L1F","L2","L3M","L3P","L4M","L4P"]:
    layer_configs[layer] = {
        "checkpoint_x" : f"checkpoints/{layer}_x.weights.h5",
        "checkpoint_y" : f"checkpoints/{layer}_y.weights.h5",
        "model_dest_x": f"checkpoints/{layer}_x.keras",
        "model_dest_y": f"checkpoints/{layer}_y.keras",
    }

# Filter rules for each layer/module type (BPIX only for now).
# Each entry is a lambda that accepts arrays (Layer, Ladder, Module)
# and returns a boolean mask selecting the matching clusters.
layer_filter_rules = {
    "L1U": lambda Layer, Ladder, Module: (Layer == 1) & (Ladder % 2 == 1),
    "L1F": lambda Layer, Ladder, Module: (Layer == 1) & (Ladder % 2 == 0),
    "L2":  lambda Layer, Ladder, Module: (Layer == 2),
    "L3M": lambda Layer, Ladder, Module: (Layer == 3) & (Module <= 4),
    "L3P": lambda Layer, Ladder, Module: (Layer == 3) & (Module >= 5),
    "L4M": lambda Layer, Ladder, Module: (Layer == 4) & (Module <= 4),
    "L4P": lambda Layer, Ladder, Module: (Layer == 4) & (Module >= 5),
}
