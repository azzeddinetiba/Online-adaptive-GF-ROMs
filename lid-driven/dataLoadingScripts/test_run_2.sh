#!/bin/bash

# Arrays of values for DENSITY and DYNAMIC_VISCOSITY
# New density values (points inside the grid but different from the reference points)
DENSITY_VALUES=(
    "5.0000E-01" "6.2500E-01" "6.8750E-01" "7.5000E-01" \
    "8.7500E-01" "1.0000E+00" "1.0625E+00" "1.1250E+00" \
    "1.2500E+00" "1.3750E+00" "1.4375E+00" "1.5000E+00" \
    "1.6250E+00" \
    "1.7500E+00" "1.8125E+00" "1.8750E+00" \
    "2.0000E+00" "2.3500E+00" \
    "1.0000E+00" \
    "6.2500E-01" "6.2500E-01" "6.2500E-01" "6.2500E-01" \
    "7.5000E-01" "7.5000E-01" "7.5000E-01" "7.5000E-01" \
    "1.0000E+00" "1.0000E+00" "1.0000E+00" "1.0000E+00" \
    "1.1250E+00" "1.1250E+00" "1.1250E+00" "1.1250E+00" \
    "1.3750E+00" "1.3750E+00" "1.3750E+00" "1.3750E+00" \
    "1.5000E+00" "1.5000E+00" "1.5000E+00" "1.5000E+00" \
    "1.7500E+00" "1.7500E+00" "1.7500E+00" "1.7500E+00" \
    "1.8750E+00" "1.8750E+00" "1.8750E+00" "1.8750E+00" \
    "5.0000E-01" "5.0000E-01" \
    "6.8750E-01" "6.8750E-01" "6.8750E-01" \
    "1.0625E+00" "1.0625E+00" "1.0625E+00" \
    "1.4375E+00" "1.4375E+00" "1.4375E+00" \
    "1.8125E+00" "1.8125E+00" "1.8125E+00" \
    "2.0000E+00" "2.0000E+00" \
    "5.0000E-01" "6.2500E-01" "6.8750E-01" "7.5000E-01" \
    "8.7500E-01" "1.0000E+00" "1.0625E+00" "1.1250E+00" \
    "1.2500E+00" "1.3750E+00" "1.4375E+00" "1.5000E+00" \
    "1.6250E+00" "1.7500E+00" "1.8125E+00" "1.8750E+00" \
    "2.0000E+00" \
    "2.3500E+00" "2.3500E+00" "2.3500E+00" "2.3500E+00" \
    "2.3500E+00" "2.3500E+00"
    "2.3500E+00" "2.3500E+00" \
    "2.3500E+00" "2.3500E+00" \
    "8.7500E-01" "8.7500E-01" \
    "1.2500E+00" "1.2500E+00" \
    "1.6250E+00" "1.6250E+00" \

)

# New viscosity values (points inside the grid but different from the reference points)
VISCOSITY_VALUES=(
    "7.0000E-03" "7.0000E-03" "7.0000E-03" "7.0000E-03" \
    "7.0000E-03" "7.0000E-03" "7.0000E-03" "7.0000E-03" \
    "7.0000E-03" "7.0000E-03" "7.0000E-03" "7.0000E-03" \
    "7.0000E-03" \
    "7.0000E-03" "7.0000E-03" "7.0000E-03" \
    "7.0000E-03" "7.0000E-03" \
    "1.0000E-02" \
    "9.0000E-03" "1.1000E-02" "1.4000E-02" "1.6000E-02" \
    "9.0000E-03" "1.1000E-02" "1.4000E-02" "1.6000E-02" \
    "9.0000E-03" "1.1000E-02" "1.4000E-02" "1.6000E-02" \
    "9.0000E-03" "1.1000E-02" "1.4000E-02" "1.6000E-02" \
    "9.0000E-03" "1.1000E-02" "1.4000E-02" "1.6000E-02" \
    "9.0000E-03" "1.1000E-02" "1.4000E-02" "1.6000E-02" \
    "9.0000E-03" "1.1000E-02" "1.4000E-02" "1.6000E-02" \
    "9.0000E-03" "1.1000E-02" "1.4000E-02" "1.6000E-02" \
    "1.0250E-02" "1.4750E-02" \
    "8.0000E-03" "1.2500E-02" "1.7000E-02" \
    "8.0000E-03" "1.2500E-02" "1.7000E-02" \
    "8.0000E-03" "1.2500E-02" "1.7000E-02" \
    "8.0000E-03" "1.2500E-02" "1.7000E-02" \
    "1.0250E-02" "1.4750E-02" \
    "1.8000E-02" "1.8000E-02" "1.8000E-02" "1.8000E-02" \
    "1.8000E-02" "1.8000E-02" "1.8000E-02" "1.8000E-02" \
    "1.8000E-02" "1.8000E-02" "1.8000E-02" "1.8000E-02" \
    "1.8000E-02" "1.8000E-02" "1.8000E-02" "1.8000E-02" \
    "1.8000E-02" \
    "8.0000E-03" "9.0000E-03" "1.0250E-02" "1.1000E-02" \
    "1.2000E-02" "1.4000E-02" \
    "1.4750E-02" "1.6000E-02" \
    "1.7000E-02" "1.8000E-02" \
    "1.0250E-02" "1.4750E-02" \
    "1.0250E-02" "1.4750E-02" \
    "1.0250E-02" "1.4750E-02" \
)

# New folder names
FOLDER_NAMES=(
    "05mu07" "06mu07" "68mu07" "07mu07" \
    "08mu07" "10mu07" "62mu07" "11mu07" \
    "12mu07" "13mu07" "14mu07" "15mu07" \
    "16mu07" \
    "17mu07" "18mu07" "87mu07" \
    "20mu07" "23mu07" \
    "10mu10" \
    "06mu09" "06mu11" "06mu14" "06mu16" \
    "07mu09" "07mu11" "07mu14" "07mu16" \
    "10mu09" "10mu11" "10mu14" "10mu16" \
    "11mu09" "11mu11" "11mu14" "11mu16" \
    "13mu09" "13mu11" "13mu14" "13mu16" \
    "15mu09" "15mu11" "15mu14" "15mu16" \
    "17mu09" "17mu11" "17mu14" "17mu16" \
    "87mu09" "87mu11" "87mu14" "87mu16" \
    "05mu10" "05mu15" \
    "68mu08" "68mu12" "68mu17" \
    "62mu08" "62mu12" "62mu17" \
    "14mu08" "14mu12" "14mu17" \
    "18mu08" "18mu12" "18mu17" \
    "20mu10" "20mu15" \
    "05mu18" "06mu18" "68mu18" "07mu18" \
    "08mu18" "10mu18" "62mu18" "11mu18" \
    "12mu18" "13mu18" "14mu18" "15mu18" \
    "16mu18" "17mu18" "18mu18" "87mu18" \
    "20mu18" \
    "23mu08" "23mu09" "23mu10" "23mu11" \
    "23mu12" "23mu14"
    "23mu15" "23mu16" \
    "23mu17" "23mu18" \
    "08mu10" "08mu15" \
    "12mu10" "12mu15" \
    "16mu10" "16mu15" \


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
    # Extract density and viscosity
    density="${DENSITY_VALUES[$i]}"
    viscosity="${VISCOSITY_VALUES[$i]}"

    # Format density for folder name (first two digits after scaling by 10)
    density_prefix=$(printf "%02d" "$(echo "$density * 10" | bc | cut -d'.' -f1)")

    # Format viscosity for folder name (first two digits after scaling by 1000)
    # Handle E-02 or E-03 by scaling appropriately
    if [[ "$viscosity" == *"E-02" ]]; then
        viscosity_suffix=$(printf "%02d" "$(echo "$viscosity * 100" | bc | cut -d'.' -f1)")
    elif [[ "$viscosity" == *"E-03" ]]; then
        viscosity_suffix=$(printf "%02d" "$(echo "$viscosity * 1000" | bc | cut -d'.' -f1)")
    else
        echo "Error: Unsupported viscosity format $viscosity"
        exit 1
    fi

    # Construct folder name
    FOLDER_NAME="${FOLDER_NAMES[$i]}"


    # Convert density and viscosity to JSON-compatible format (lowercase "e")
    density_json=$(echo "$density" | sed 's/E/e/g')
    viscosity_json=$(echo "$viscosity" | sed 's/E/e/g')

    # Modify the JSON file
    sed -i '' "s/\"param_0_value\"[[:space:]]*:[[:space:]]*[0-9.e+-]*,/\"param_0_value\": ${density_json},/" "$COSIM_FILE"
    sed -i '' "s/\"param_1_value\"[[:space:]]*:[[:space:]]*[0-9.e+-]*/\"param_1_value\": ${viscosity_json}/" "$COSIM_FILE"

    # Update the values in the .mdpa file
    sed -i '' "10s/.*/    DENSITY   ${density}/" "$MDPA_FILE"
    sed -i '' "11s/.*/    DYNAMIC_VISCOSITY   ${viscosity}/" "$MDPA_FILE"

    # Run the Python script using the full path to pythonK
    echo "Launching the Simu."
    zsh -c "source ~/.zshrc && /opt/homebrew/bin/python3 MainKratos.py"

    # Check if the result folder exists
    RESULT_FOLDER="coSimData"
    if [ -d "$RESULT_FOLDER" ]; then
        # Delete all files containing "data" in their name within the coSimData folder
        find "$RESULT_FOLDER" -type f -name "*data*" -exec rm -f {} \;

        # Create the new folder and move the coSimData folder into it
        mkdir -p "$FOLDER_NAME"
        mv "$RESULT_FOLDER" "$FOLDER_NAME/"
        echo "Results moved to $FOLDER_NAME/$RESULT_FOLDER"
    else
        echo "Error: Result folder '$RESULT_FOLDER' not found."
    fi
done
