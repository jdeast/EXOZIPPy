import yaml


def save_mass_matrix(idata, model, output_path):
    # `model` supplies the free_RV ordering the variances are indexed by, so it
    # has to be passed in -- it used to be read from a name that was never
    # defined here, which would have raised NameError on the first call. There
    # are no call sites yet; add the model argument when you write one.

    # Extract the final learned scales from the warmup info
    # This is the diagonal of the inverse mass matrix
    learned_variances = (
        idata.warmup_posterior_adaptive_info.model_logp_scaling.values[-1]
    )

    # Map them to the free_RV names to ensure unique IDs
    # Using 'model.free_RVs' ensures we match the sampler's order
    mapping = {
        var.name: float(learned_variances[i])
        for i, var in enumerate(model.free_RVs)
    }

    with open(output_path, "w") as f:
        yaml.dump(mapping, f)
