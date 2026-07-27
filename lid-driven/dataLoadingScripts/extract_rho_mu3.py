def extract_density_viscosity3(folder_names):
    # Mapping of folder prefixes to actual density and viscosity values
    density_map = {
        "05": 0.5000,
        "06": 0.6250,
        "68": 0.6875,
        "07": 0.7500,
        "08": 0.8750,
        "10": 1.0000,
        "62": 1.0625,
        "11": 1.1250,
        "12": 1.2500,
        "13": 1.3750,
        "14": 1.4375,
        "15": 1.5000,
        "16": 1.6250,
        "17": 1.7500,
        "18": 1.8125,
        "87": 1.8750,
        "20": 2.0000,
        "23": 2.3500
    }

    viscosity_map = {
        "07": 0.7000e-2,
        "08": 0.8000e-2,
        "09": 0.9000e-2,
        "10": 1.0250e-2,
        "11": 1.1000e-2,
        "12": 1.2500e-2,
        "14": 1.4000e-2,
        "15": 1.4750e-2,
        "16": 1.6000e-2,
        "17": 1.7000e-2,
        "18": 1.8000e-2
    }

    # Initialize arrays for density and viscosity
    density_values = []
    viscosity_values = []

    # Extract density and viscosity from folder names
    for folder in folder_names:
        density_prefix = folder[:2]  # First two characters for density
        viscosity_suffix = folder[-2:]  # Last two characters for viscosity

        # Map to actual values
        density_values.append(density_map[density_prefix])
        viscosity_values.append(viscosity_map[viscosity_suffix])

    return density_values, viscosity_values
