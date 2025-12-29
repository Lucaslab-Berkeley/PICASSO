import picasso


def test_imports_with_version():
    assert isinstance(picasso.__version__, str)
