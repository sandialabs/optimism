import yaml


def parse_yaml_input_file(input_file: str) -> dict:
    with open(input_file, 'rb') as f:
        try:
            d = yaml.safe_load(f)
        except yaml.YAMLError:
            raise ValueError('Issue with yaml input file yaml format!')
    return d
