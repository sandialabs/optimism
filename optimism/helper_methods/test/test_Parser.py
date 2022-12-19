import os
import pytest

from optimism.helper_methods.Parser import dump_input_file
from optimism.helper_methods.Parser import parse_yaml_input_file


def test_parse_yaml_file():
    yaml_file = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'good_yaml_file.yaml')
    d = parse_yaml_input_file(yaml_file)
    assert d['mesh']['type'] == 'exodus mesh'


def test_bad_yaml_file_raise():
    yaml_file = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'bad_yaml_file.yaml')
    with pytest.raises(ValueError):
        parse_yaml_input_file(yaml_file)


def test_dump_input_file():
    yaml_file = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'good_yaml_file.yaml')
    dump_input_file(yaml_file)
