import  yaml

def save_mass_matrix(idata, output_path):
    # Extract the final learned scales from the warmup info
    # This is the diagonal of the inverse mass matrix
    learned_variances = idata.warmup_posterior_adaptive_info.model_logp_scaling.values[-1]

    # Map them to the free_RV names to ensure unique IDs
    # Using 'model.free_RVs' ensures we match the sampler's order
    mapping = {var.name: float(learned_variances[i]) for i, var in enumerate(model.free_RVs)}

    with open(output_path, 'w') as f:
        yaml.dump(mapping, f)