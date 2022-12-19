from .MaterialModel import MaterialModel


class MaterialModelNameError(Exception): pass


def material_model_factory(model_name: str, props: dict) -> MaterialModel:
    if model_name.lower() == 'gent':
        from .Gent import create_material_model_functions
    elif model_name.lower() == 'linear elastic':
        from .LinearElastic import create_material_model_functions
    elif model_name.lower() == 'j2plastic':
        from .J2Plastic import create_material_model_functions
    elif model_name.lower() == 'neohookean':
        from .Neohookean import create_material_model_functions
    else:
        print('\n\n')
        print('Invalid material model name "%s"\n' % model_name)
        raise MaterialModelNameError
    return create_material_model_functions(props)
