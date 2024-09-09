import pandas as pd

# Load the tracking data
tracking_data = pd.read_csv('Sample_Game_1_RawTrackingData_Away_Team.csv')

# Function to identify final passes
def identify_final_passes(tracking_data):
    final_passes = []
    
    # Iterate through the events to identify passes and goal attempts
    for i in range(1, len(tracking_data)):
        event = tracking_data.iloc[i]
        previous_event = tracking_data.iloc[i-1]
        
        # Detect a goal attempt (shot on goal)
        if event['event_type'] == 'shot':
            shooter_id = event['player_id']
            shot_time = event['timestamp']
            
            # Trace back to find the last pass to this shooter
            for j in range(i-1, 0, -1):
                pass_event = tracking_data.iloc[j]
                
                if pass_event['event_type'] == 'pass' and pass_event['receiver_id'] == shooter_id:
                    pass_time = pass_event['timestamp']
                    
                    # Check spatial and temporal proximity
                    if (shot_time - pass_time) < 10:  # assuming time difference in seconds
                        final_passes.append(pass_event)
                        break

    return pd.DataFrame(final_passes)

# Identify final passes
final_passes = identify_final_passes(tracking_data)

# Output the results
print("Identified final passes:")
print(final_passes)
