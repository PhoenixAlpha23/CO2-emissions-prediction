import pandas as pd

def adjust_weight_by_stops(weight, num_stops, reduction_factor):
    adjusted_weight = weight * (1 - reduction_factor * num_stops)
    return adjusted_weight

def calculate_emissions(initial_weight, distances, num_stops):
    total_emissions = 0
    adjusted_weight = initial_weight
    
    for i in range(num_stops):
        reduction_factor = (adjusted_weight + 1) / (2 * initial_weight)
        adjusted_weight = adjust_weight_by_stops(adjusted_weight, 1, reduction_factor)
        leg_emissions = adjusted_weight * distances[i] * emission_factor
        total_emissions += leg_emissions
        
    return total_emissions

def predict_emissions_with_cargo(model, vehicle_class, engine_size, fuel_type, 
                                fuel_consumption, weight, num_stops, distances):
    """
    Predicts CO2 emissions for cargo vehicles with adjustable weight impact.
    
    Args:
        model: Trained model
        vehicle_class (str): Type of vehicle
        engine_size (float): Engine size in liters
        fuel_type (str): Type of fuel
        fuel_consumption (float): Fuel consumption in L/100 km
        weight (float): Initial cargo weight in kg
        num_stops (int): Number of delivery stops
        distances (list): Distance in km for each delivery stop
    
    Returns:
        float: Predicted total CO2 emissions in g
    
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
    
    if not isinstance(num_stops, int) or num_stops <= 0:
        raise ValueError("Invalid number of stops. Number of stops should be a positive integer.")
    
    if not isinstance(distances, list) or len(distances) != num_stops:
        raise ValueError("Invalid distances. Distances should be a list with length equal to the number of stops.")
    
    # Create input data
    input_data = pd.DataFrame({
        'Vehicle Class': [vehicle_class],
        'Engine Size(L)': [engine_size],
        'Fuel Type': [fuel_type],
        'Fuel Consumption Comb (L/100 km)': [fuel_consumption]
        'Distances': [distances],
        'Number of Stops': [num_stops]
    })
    
    # Predict base emissions
    base_emissions = model.predict(input_data)[0]
    
    # Calculate total emissions with weight adjustment
    total_emissions = calculate_emissions(weight, distances, num_stops)
    
    return total_emissions
