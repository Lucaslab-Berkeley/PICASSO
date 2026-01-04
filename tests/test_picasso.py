"""Basic tests for PICASSO package."""

import picasso


def test_imports_with_version():
    """Test that package can be imported and has a version."""
    assert isinstance(picasso.__version__, str)
    assert len(picasso.__version__) > 0


def test_import_core():
    """Test that core module can be imported."""
    from picasso.core import core_refine_ctf

    assert callable(core_refine_ctf)


def test_import_config():
    """Test that config modules can be imported."""
    from picasso.config import (
        ComputationalConfig,
        CtfRefineConfig,
        InputConfig,
        OutputConfig,
    )

    assert CtfRefineConfig is not None
    assert OutputConfig is not None
    assert ComputationalConfig is not None
    assert InputConfig is not None


def test_import_utils():
    """Test that utility modules can be imported."""
    from picasso.utils.data_io import (
        read_refined_ctf_params,
        write_loss_csvs,
        write_refined_ctf_params,
    )

    assert callable(read_refined_ctf_params)
    assert callable(write_refined_ctf_params)
    assert callable(write_loss_csvs)
