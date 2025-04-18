begin density_topology
    mesh_name EXO_files/ellipse_test_Seeded.exo
    output_name to-result.exo
    initial_density_value 0.4
end

begin kernel_filter
    centering_type element
    filter_radius 2.0
    use_relative_radius true
end

begin objective strain_energy
    app plato-python-app
    criterion plato-python-app
    input_files plato-python-app-input.xml
    aggregation_weight 1.0
end

begin rol_optimization
    max_iterations 100
end

begin gradient_check
    output_file_name ROL_gradient_check_output.txt
    number_of_steps 10
    initial_direction_magnitude 1
    step_size_reduction_factor 0.1
    random_direction_seed 123
end
