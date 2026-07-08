from .parameter import Parameter
from ..physics_registry import PHYSICS_REGISTRY
from abc import ABC, abstractmethod
import numpy as np
import pytensor.tensor as pt


class Component(ABC):
    """
    Base class for all physical and instrumental components in the system.

    This framework utilizes a "Lazy DAG" (Directed Acyclic Graph) architecture to
    safely construct complex PyMC models without deadlocks. The orchestration
    happens in the following distinct lifecycle stages:

    Stage 0: load_data()           - Ingests CSVs and calculates data-driven parameter estimates.
    Stage 1: build_maps()          - Generates Numpy integer arrays linking children to parents.
    Stage 2: register_parameters() - Declares the component's mathematical manifest.
    Stage 3: [System-Level]        - The ConfigManager symbolically solves the universe.
    Stage 4: build_tensor_maps()   - Auto-converts Numpy maps to PyTensor variables.
    Stage 5: add_parameter()       - Materializes PyMC nodes safely, one at a time.
    Stage 6: build_likelihood()    - Defines observational Likelihoods and Potentials.
    """

    def __init__(self, component_config, config_manager):
        """Standardized constructor for ALL components."""
        self.config = component_config
        self.config_manager = config_manager

        # Determine how many of this thing we are building
        self.n_elements = len(self.config)

        # Grab names for labeling PyMC nodes
        self.names = [c.get("name", f"{i}") for i, c in enumerate(self.config)]

        # Enforce unique names
        if len(set(self.names)) != len(self.names):
            raise ValueError(
                f"Duplicate names found in {self.__class__.__name__} configuration: {self.names}. "
                f"All component names must be unique."
            )

    @property
    @abstractmethod
    def prefix(self):
        """Naming prefix for the model (e.g., 'star', 'planet', 'inst')."""
        pass

    def load_data(self, system):
        """
        Stage 1a: Data Ingestion.
        Override this to load CSV files and push data-driven parameter guesses (like RV offsets)
        to the ConfigManager.
        """
        pass

    def build_maps(self):
        """
        Stage 1b: Logical Mapping.
        Override this to define Numpy integer arrays (ending in '_map') that establish
        vectorized relationships between this component and its parents.
        """
        pass

    @abstractmethod
    def register_parameters(self, system):
        """
        Stage 2: The Blueprint.
        Define `self.manifest` (a dictionary) mapping parameter names to their physics
        dependencies, and push those symbols to the ConfigManager.
        """
        pass

    def build_tensor_maps(self):
        """
        Stage 4: Automatic PyTensor Conversion.
        Scans the component's attributes. Any numpy array ending in '_map'
        is automatically converted to a PyTensor variable ending in '_map_tensor'.
        """
        for attr_name in list(self.__dict__.keys()):
            if attr_name.endswith("_map"):
                logical_array = getattr(self, attr_name)
                # Only convert if it's actually an array/list (safeguard)
                if isinstance(logical_array, (np.ndarray, list)):
                    tensor_name = attr_name + "_tensor"
                    tensor_var = pt.as_tensor_variable(logical_array).astype("int32")
                    setattr(self, tensor_name, tensor_var)

    def add_parameter(self, model, param_name, system, context_nodes=None):
        context_nodes = context_nodes or {}

        # 0. Prevent double-building nodes
        if hasattr(self, param_name) and isinstance(getattr(self, param_name), Parameter):
            return getattr(self, param_name).value

        if not hasattr(self, 'manifest'):
            raise ValueError(f"[{self.prefix}] has no manifest. Did register_parameters run?")
        if param_name not in self.manifest:
            raise KeyError(f"[{self.prefix}] System requested '{param_name}', but it is not in the manifest.")

        options = self.manifest[param_name] or {}
        if isinstance(options, str):
            options = {"expr_key": options}
        options = dict(options)  # don't mutate the manifest via the pops below

        # Manifest entries may override the shape for parameters that are not
        # one-per-element (e.g. one (s, alpha) per lens companion), and the
        # per-element names used for user-param resolution and display labels
        # (e.g. per-source lens params named after the source stars).
        shape = tuple(options.pop("shape", None) or (self.n_elements,))
        names = options.pop("names", None) or getattr(self, 'names', None)

        # 1. Grab configuration properties agnostically
        cfg = self.config_manager.resolve(
            self.prefix, param_name, shape=shape, names=names
        )

        expr_key = options.pop("expr_key", None)
        expressions_dict = cfg.pop("expressions", {})
        expression = None

        # --- AGNOSTIC CONDITIONAL WIRE-UP ---
        # Only parse dependencies if an expression block actively exists for this parameter role
        if expr_key and expr_key in expressions_dict:
            expr_cfg = expressions_dict[expr_key]
            func_name = expr_cfg.get("func_name")

            if func_name not in PHYSICS_REGISTRY:
                raise NotImplementedError(
                    f"[{self.prefix}.{param_name}] Function '{func_name}' not in PHYSICS_REGISTRY.")

            func = PHYSICS_REGISTRY[func_name]
            manifest_deps = options.pop("deps", None)
            dep_names = manifest_deps if manifest_deps is not None else expr_cfg.get("deps", [])
            dep_nodes = []

            for d in dep_names:
                if d in context_nodes:
                    dep_nodes.append(context_nodes[d])
                elif "." in d:
                    # Parse universal cross-component strings: "star.density[star_map]"
                    custom_slice = None
                    if "[" in d and d.endswith("]"):
                        path_part, slice_part = d.split("[", 1)
                        custom_slice = slice_part.rstrip("]")
                        d_lookup = path_part
                    else:
                        d_lookup = d

                    ext_comp_name, ext_param_name = d_lookup.split(".", 1)
                    ext_comp = getattr(system, ext_comp_name, None)
                    if not ext_comp:
                        raise ValueError(f"[{self.prefix}.{param_name}] Component '{ext_comp_name}' is not active.")

                    # Ensure the dependency node is built lazily on demand
                    if not hasattr(ext_comp, ext_param_name):
                        ext_comp.add_parameter(model, ext_param_name, system, context_nodes)

                    ext_param = getattr(ext_comp, ext_param_name)

                    # Dynamically slice via requested map name or component fallback name
                    map_attr = f"{custom_slice}_tensor" if custom_slice else f"{ext_comp_name}_map_tensor"
                    if hasattr(self, map_attr):
                        map_tensor = getattr(self, map_attr)
                        dep_nodes.append(ext_param.value[map_tensor])
                    else:
                        dep_nodes.append(ext_param.value)
                else:
                    # Local tracking recursive lookup
                    if not hasattr(self, d) or not isinstance(getattr(self, d), Parameter):
                        self.add_parameter(model, d, system, context_nodes)
                    dep_nodes.append(getattr(self, d).value)

            expression = lambda: func(*dep_nodes)

        # 2b. Wire up user-defined parameter links (initval/mu/lower/upper
        # expressions from the params file referencing other parameters).
        element_links = self._wire_user_links(model, param_name, system, cfg, expression)

        # 3. Create Parameter Node
        full_params = {**cfg, **options}
        param_obj = Parameter(
            label=f"{self.prefix}.{param_name}",
            names=names,
            expression=expression,
            element_links=element_links,
            user_params=self.config_manager.user_params,
            **full_params
        )

        setattr(self, param_name, param_obj)
        return param_obj.build_pymc()

    def _wire_user_links(self, model, param_name, system, cfg, expression):
        """
        Translate the ConfigManager's user-defined links targeting this
        parameter into per-element PyTensor closures for Parameter.build_pymc.

        Each closure receives this parameter's own physical vector (internal
        units) so same-parameter references (star.A.age -> star.B.age) resolve
        without leaving the node; external references are materialized lazily
        through add_parameter, exactly like physics expression dependencies.
        Unit convention: referenced parameters contribute their values in
        their own user units; the result is taken in the target's user unit.
        """
        cm = self.config_manager
        get_links = getattr(cm, "get_element_links", None)
        if get_links is None:
            return None
        links = get_links(self.prefix, param_name)
        if not links:
            return None

        from ..linking import sympy_to_pytensor

        sigma_arr = cfg.get("sigma")
        out = {}
        for fld, per_elem in links.items():
            for idx, plink in per_elem.items():
                # Classify the runtime role of this link.
                if fld == "initval":
                    s = sigma_arr[idx] if sigma_arr is not None else np.nan
                    if s == 0:
                        key = "hard"     # derived element: tracks the expression exactly
                    elif s > 0:
                        key = "mu"       # soft link: Gaussian penalty on the difference
                    else:
                        continue         # initialization-only; solver already applied it
                elif fld == "mu":
                    key = "mu"
                elif fld in ("lower", "upper"):
                    key = fld
                else:
                    continue             # sigma / init_scale: static snapshots

                if expression is not None and key != "mu":
                    raise ValueError(
                        f"[{self.prefix}.{param_name}] link '{plink.expr_str}' targets "
                        f"field '{fld}', but this parameter is derived from a physics "
                        f"expression; only soft (mu) links are supported there.")

                ext_vals = {}     # dep path -> tensor in the dep's USER units
                self_refs = {}    # dep path -> (element index, user->internal factor)
                for dep in plink.dep_paths:
                    dparts = dep.split('.')
                    dcomp, didx, dparam = dparts[0], int(dparts[1]), dparts[2]
                    dfactor = cm.get_conversion_factor(dcomp, dparam, full_path=dep)
                    if dcomp == self.prefix and dparam == param_name:
                        self_refs[dep] = (didx, dfactor)
                        continue
                    comp = self if dcomp == self.prefix else getattr(system, dcomp, None)
                    if comp is None:
                        raise ValueError(
                            f"[{self.prefix}.{param_name}] link '{plink.expr_str}' "
                            f"references component '{dcomp}', which is not active.")
                    if not (hasattr(comp, dparam) and isinstance(getattr(comp, dparam), Parameter)):
                        comp.add_parameter(model, dparam, system)
                    node = getattr(comp, dparam).value
                    if getattr(node, 'ndim', 0) >= 1:
                        node = node[didx]
                    elif didx != 0:
                        raise ValueError(
                            f"[{self.prefix}.{param_name}] link '{plink.expr_str}': "
                            f"'{dep}' indexes element {didx} of a scalar parameter.")
                    ext_vals[dep] = node / dfactor if dfactor != 1.0 else node

                tfactor = cm.get_conversion_factor(
                    self.prefix, param_name,
                    full_path=f"{self.prefix}.{idx}.{param_name}")

                def make_fn(plink=plink, ext_vals=ext_vals, self_refs=self_refs,
                            tfactor=tfactor):
                    def fn(phys_internal):
                        vals = dict(ext_vals)
                        for dep, (j, f) in self_refs.items():
                            v = phys_internal[j]
                            vals[dep] = v / f if f != 1.0 else v
                        user_val = sympy_to_pytensor(plink.expr, vals)
                        return user_val * tfactor if tfactor != 1.0 else user_val
                    return fn

                out.setdefault(key, {})[idx] = {
                    "fn": make_fn(),
                    "intra_deps": {j for (j, _) in self_refs.values()},
                }

        return out or None

    @abstractmethod
    def build_likelihood(self, model, system):
        """
        Stage 6: The Objective Function.
        Construct the PyMC Likelihoods (`pm.Normal`, etc.) or custom `pm.Potential`
        penalties that constrain the model against data.
        """
        pass

    def compile_plotters(self, model, system):
        """
        Compile fast PyTensor functions for plotting.
        Translates PyTensor graphs into numpy functions to ensure consistency
        between the likelihood calculation and the final figures.
        """
        pass

    def plot(self, system, points, filename_prefix="debug"):
        """
        Plot the model and data. Called twice:
          - Pre-flight: To visually verify the initialization logic.
          - Post-flight: To generate publication-quality posterior models.
        """
        pass

    def sampler_requirements(self):
        """Return sampler constraints imposed by this component.

        Returns a dict with optional keys:
          'incompatible' : set of method names that cannot be used
          'recommended'  : preferred method name (str)
          'reason'       : human-readable explanation for warnings (str)

        The default implementation returns no constraints.  Override when a
        component uses non-differentiable Ops (e.g. MulensModel) that are
        incompatible with gradient-based samplers.
        """
        return {}

    def _is_sampling_param(self, attr):
        """Helper to identify parameters that need to be passed to the compiled function."""
        return isinstance(attr, Parameter) and attr.expression is None

    def get_parameters(self, sampling_only=False):
        """
        Returns all Parameter objects belonging to this component.
        If sampling_only=True, filters for Free (sampled) parameters.
        """
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Parameter):
                if sampling_only:
                    if attr.expression is None:
                        params.append(attr)
                else:
                    params.append(attr)
        return params