from optimism import Parser


if __name__ == '__main__':
    yaml_file = 'lattice_geometry.yaml'
    Parser.setup_simulation_from_input_file(yaml_file)
