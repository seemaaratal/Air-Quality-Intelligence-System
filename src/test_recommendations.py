def get_recommendation(aqi):

    if aqi <= 50:
        return "Good - Continue regular monitoring."

    elif aqi <= 100:
        return "Satisfactory - Monitor traffic and pollution."

    elif aqi <= 200:
        return "Pollution Hotspot - Control dust and traffic emissions."

    elif aqi <= 300:
        return "Red Zone - Issue alerts and strengthen pollution control."

    elif aqi <= 400:
        return "Very Poor - Apply strict pollution-control measures."

    else:
        return "Severe - Immediate action and continuous monitoring required."


test_values = [40, 80, 150, 250, 350, 450]


print("\nRECOMMENDATION ENGINE TEST\n")

for aqi in test_values:

    print(
        "AQI:",
        aqi,
        "→",
        get_recommendation(aqi)
    )