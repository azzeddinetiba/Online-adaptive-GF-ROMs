#!/bin/bash

# Arrays of values for DENSITY and DYNAMIC_VISCOSITY
# Density values spanning from 0.5 to 2.0 with 5 points
DENSITY_VALUES=("5.0000E-01" "5.0000E-01" "5.0000E-01"\
                "8.7500E-01" "8.7500E-01" "8.7500E-01"\
                "1.2500E+00" "1.2500E+00" "1.2500E+00"\
                "1.6250E+00" "1.6250E+00" "1.6250E+00"\
                "2.0000E+00" "2.0000E+00" "2.0000E+00")

# Viscosity values spanning from 0.8e-2 to 1.7e-2 with 3 points
VISCOSITY_VALUES=("8.0000E-03" "1.2500E-02" "1.7000E-02"\
                  "8.0000E-03" "1.2500E-02" "1.7000E-02"\
                  "8.0000E-03" "1.2500E-02" "1.7000E-02"
                  "8.0000E-03" "1.2500E-02" "1.7000E-02"
                  "8.0000E-03" "1.2500E-02" "1.7000E-02")

# Folder names generated based on the density and viscosity grid
FOLDER_NAMES=(
    "05mu08" "05mu12" "05mu17"
    "08mu08" "08mu12" "08mu17"
    "12mu08" "12mu12" "12mu17"
    "16mu08" "16mu12" "16mu17"
    "20mu08" "20mu12" "20mu17"
)


# File to modify
MDPA_FILE="fsi_lid_driven_cavity_Fluid.mdpa"
COSIM_FILE="ProjectParametersCoSim.json"

# Check if the arrays have the same length
if [ "${#DENSITY_VALUES[@]}" -ne "${#VISCOSITY_VALUES[@]}" ]; then
    echo "Error: DENSITY and VISCOSITY arrays must have the same length."
    exit 1
fi
if [ "${#DENSITY_VALUES[@]}" -ne "${#FOLDER_NAMES[@]}" ]; then
    echo "Error: DENSITY and VISCOSITY arrays must have the same length."
    exit 1
fi

# Iterate over the arrays
for i in "${!DENSITY_VALUES[@]}"; do
    DENSITY="${DENSITY_VALUES[$i]}"
    VISCOSITY="${VISCOSITY_VALUES[$i]}"

    FOLDER_NAME="${FOLDER_NAMES[$i]}"

    # Convert scientific notation to lowercase "e" for JSON compatibility
    DENSITY_JSON=$(echo "$DENSITY" | sed 's/E/e/g')
    VISCOSITY_JSON=$(echo "$VISCOSITY" | sed 's/E-02//g')  # Remove "e-02" from viscosity

    # Update the values in the .mdpa file
    sed -i '' "10s/.*/    DENSITY   ${DENSITY}/" "$MDPA_FILE"
    sed -i '' "11s/.*/    DYNAMIC_VISCOSITY   ${VISCOSITY}/" "$MDPA_FILE"

    # Update the values of param_0_value and param_1_value in the JSON file
    sed -i '' "s/\"param_0_value\"[[:space:]]*:[[:space:]]*[0-9.e+-]*,/\"param_0_value\": ${DENSITY_JSON},/" "$COSIM_FILE"
    sed -i '' "s/\"param_1_value\"[[:space:]]*:[[:space:]]*[0-9.e+-]*/\"param_1_value\": ${VISCOSITY_JSON}/" "$COSIM_FILE"

    # Run the Python script using the full path to pythonK
    echo "Launching the Simu."
    zsh -c "source ~/.zshrc && /opt/homebrew/bin/python3 MainKratos.py"

    # Check if the result folder exists
    RESULT_FOLDER="coSimData"
    if [ -d "$RESULT_FOLDER" ]; then
        # Delete all files containing "data" in their name within the coSimData folder
        # find "$RESULT_FOLDER" -type f -name "*data*" -exec rm -f {} \;

        # Create the new folder and move the coSimData folder into it
        mkdir -p "$FOLDER_NAME"
        mv "$RESULT_FOLDER" "$FOLDER_NAME/"
        echo "Results moved to $FOLDER_NAME/$RESULT_FOLDER"
    else
        echo "Error: Result folder '$RESULT_FOLDER' not found."
    fi
done
