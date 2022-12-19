import re
import yaml


def parse_yaml_input_file(input_file: str) -> dict:
    with open(input_file, 'rb') as f:
        try:
            d = yaml.safe_load(f)
        except yaml.YAMLError:
            raise ValueError('Issue with yaml input file yaml format!')
    return d


def dump_input_file(input_file: str) -> None:
    with open(input_file, 'r') as f:
        lines = f.readlines()

    print('######################################################################')
    print('# BEGIN INPUT FILE DUMP')
    print('######################################################################\n')
    for line in lines:
        print(re.sub('\n', '', line))
    print()
    print('######################################################################')
    print('# END INPUT FILE DUMP')
    print('######################################################################\n')