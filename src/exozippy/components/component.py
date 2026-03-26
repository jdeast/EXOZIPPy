from typing import Dict, Any
from .parameter import Parameter


class Component:
    """Parent class for all physical model components (Star, Planet, Orbit, etc.)"""

    def get_parameter_lines(self):
        """Recursively gather (variables_tex, table_line_tex) from all child Parameters."""
        var_lines = []
        table_lines = []

        # 1. Look for Parameters directly on this component
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Parameter):
                var_lines.append(attr.to_latex_var())
                table_lines.append(attr.to_table_line())

        # 2. Recurse into child Components (like Planet -> Orbit)
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Component):
                v, t = attr.get_parameter_lines()
                var_lines.extend(v)
                table_lines.extend(t)

        return var_lines, table_lines

    def get_latex_data(self):
        defs = []
        lines = []

        # Use __dict__ to preserve the order you wrote in the class
        for attr_name, attr in self.__dict__.items():
            if isinstance(attr, Parameter):
                if attr.print_to_table:
                    defs.append(attr.to_latex_def())
                    lines.append(attr.to_table_line())

            # Also preserve order for sub-components (like Orbit)
            elif isinstance(attr, Component):
                child_defs, child_lines = attr.get_latex_data()
                defs.extend(child_defs)
                lines.extend(child_lines)

        return defs, lines

    def get_latex_data_old(self):
        """Recursively gather (variable_defs, table_lines) from all Parameters."""
        defs = []
        lines = []

        # 1. Gather from immediate Parameter attributes
        # Sort by attr_name or a 'priority' field to keep table order consistent
        for attr_name in sorted(dir(self)):
            attr = getattr(self, attr_name)
            if isinstance(attr, Parameter):
                if attr.print_to_table:
                    defs.append(attr.to_latex_def())  # e.g., \newcommand{\ezRstar}{1.0 \pm 0.1}
                    lines.append(attr.to_table_line())  # e.g., R_* & [Radius] & \ezRstar \\

        # 2. Recurse into children (Orbit, etc.)
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Component):
                child_defs, child_lines = attr.get_latex_data()
                defs.extend(child_defs)
                lines.extend(child_lines)

        return defs, lines

    def get_inits(self) -> Dict[str, float]:
        inits = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Parameter) and attr.expression is None:
                # Use the nudged initval we calculated in build_pymc
                inits[attr.label] = attr.initval

            if isinstance(attr, Component):
                inits.update(attr.get_inits())
        return inits

    def get_scales(self) -> Dict[str, float]:
        """
        Automatically crawls the component's attributes to find all sampling
        parameters and returns their initial scales.
        """
        scales = {}
        for attr_name in dir(self):
            # We skip 'internal' dunder methods for speed
            if attr_name.startswith('__'):
                continue

            attr = getattr(self, attr_name)

            # Check if it's a Parameter object
            if isinstance(attr, Parameter):
                # We only want to scale 'Free Variables' (no expression)
                if attr.expression is None:
                    if attr.init_scale is not None:
                        scales[attr.label] = attr.init_scale
                    else:
                        # Warning with context so you know which component is 'lazy'
                        print(f"WARNING: No init_scale for {attr.label} "
                              f"in {self.__class__.__name__} '{self.name}'. Defaulting to 1.0.")
                        scales[attr.label] = 1.0

        # Now, handle nested components (like Planet owning an Orbit)
        # This recursively gathers scales down the tree
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Component):
                scales.update(attr.get_scales())

        return scales