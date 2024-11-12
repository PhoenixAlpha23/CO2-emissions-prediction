import pandas as pd

def predict_emissions_with_cargo(model, vehicle_class, engine_size, fuel_type, 
                               fuel_consumption, weight):
    """
    Predicts CO2 emissions for cargo vehicles with adjustable weight impact.
    
    Args:
        model: Trained model
        vehicle_class (str): Type of vehicle
        engine_size (float): Engine size in liters
        fuel_type (str): Type of fuel
        fuel_consumption (float): Fuel consumption in L/100 km
        weight (float): Cargo weight in kg
    
    Returns:
        float: Predicted CO2 emissions in g/km
    
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    valid_vehicle_classes = ['Cargo Van', 'Pickup Truck', 'SUV - STANDARD']
    if vehicle_class not in valid_vehicle_classes:
        raise ValueError(
            f"Invalid vehicle class. Must be one of {valid_vehicle_classes}")
    
    if not isinstance(weight, (int, float)) or weight <= 0:
        raise ValueError("Invalid weight. Weight should be a positive number.")

    # Create input data
    input_data = pd.DataFrame({
        'Vehicle Class': [vehicle_class],
        'Engine Size(L)': [engine_size],
        'Fuel Type': [fuel_type],
        'Fuel Consumption Comb (L/100 km)': [fuel_consumption]
    })

    # Predict base emissions
    base_emissions = model.predict(input_data)[0]
    
    # Adjust emissions based on weight (0.073 g/km per kg)
    weight_adjustment = weight * 0.073
    
    return base_emissions + weight_adjustment
