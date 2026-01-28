# Docstrings

Use this style:

        """
        Map a dict[filename: value] to dict[dataset: value].

        Parameters
        ----------
        filename_dict : dict or None
            Keys are filenames, values are bool

        Returns
        -------
        dict
            Keys are MulensData objects, values from filename_dict
        """


# Don't use type hints

Do this:

def __init__(
            self,
            files=None,
            datasets=None,
            fit_type=None)

NOT this:

def __init__(
            self,
            files: list=None,
            datasets: list=None,
            fit_type: str=None)
