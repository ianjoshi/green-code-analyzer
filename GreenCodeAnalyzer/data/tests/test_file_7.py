import pandas as pd
import json

def add_fim_markers(prefix: str, suffix: str = "") -> str:

    fim_text = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"

    return fim_text

def parse_measurements(data):
    try:
        measurement_dict = json.loads(data.replace("'", "\""))
        return measurement_dict
    except (ValueError, TypeError):
        print("Error in parse_measurements")
        return {}



def generate_completion_trigger_data(CUPS):
    data = []
    document = "Untitled-1"


    parsed_measurements = CUPS['Measurements'].apply(parse_measurements)
    measurements_df = pd.json_normalize(parsed_measurements)
    #save measurements_df as a csv file
    measurements_df.to_csv('Project\Data\dataLabeling\measurements_df.csv', index=False)

    for i, row in CUPS.iterrows():
        text_change = row["CurrentSuggestion"]
        char_amount = measurements_df["documentLength"][i] + measurements_df["promptCharLen"][i]
        line_count = measurements_df["numLines"][i]
        char_count_before = measurements_df["documentLength"][i]
        char_count_after = 0
        text_before = row["CurrentPrompt"]
        text_after = ""
        time_since_last_change = measurements_df["timeSinceIssuedMs"][i]

        data_point = {
            "text_change": text_change, 
            "document": document, 
            "line_count": line_count, 
            "char_amount": char_amount, 
            "char_count_before": char_count_before, 
            "char_count_after": char_count_after, 
            "text_before": text_before, 
            "text_after": text_after, 
            "time_since_last_change": time_since_last_change 
        }

        data.append(data_point)
    
    return data





def main():

   
    CUPS = pd.read_csv('Project\Data\dataLabeling\majority_voted_dataset.csv')

    output_file = 'Project\Data\CUPS_completion_trigger_data.json'
    data = generate_completion_trigger_data(CUPS)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Generated {len(data)} completion trigger data points and saved to {output_file}")

if __name__ == "__main__":
    main()



