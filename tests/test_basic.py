"""
Basic tests to ensure testing framework is working
"""

import pytest


def test_basic_assertion():
    """Test that basic assertions work"""
    assert True
    assert 1 + 1 == 2


def test_import_main_package():
    """Test that main package can be imported"""
    # For now, just check that we can import built-in modules
    import sys
    import os

    assert sys.version_info >= (3, 11)
    assert os.path.exists(__file__)


class TestBasicClass:
    """Test that class-based tests work"""

    def test_class_method(self):
        """Test method in test class"""
        result = "hello"
        assert result == "hello"

    def test_arithmetic(self):
        """Test basic arithmetic"""
        assert 2 * 3 == 6
        assert 10 / 2 == 5


@pytest.mark.parametrize(
    "input,expected",
    [
        (1, 1),
        (2, 4),
        (3, 9),
        (4, 16),
    ],
)
def test_parametrized(input, expected):
    """Test parametrized tests work"""
    assert input**2 == expected
