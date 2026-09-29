"""Dask is optional for serial optimization and external ask/tell workers."""
try:
    from distributed import Client
except (ImportError, OSError) as exc:
    _dask_import_error = str(exc)

    class Client:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Dask is unavailable in this environment: " + _dask_import_error
                + ". Use n_workers=1 or distribute ask()/tell() externally."
            )
