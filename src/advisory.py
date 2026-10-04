"""Rule-based heat advisories, general and per audience.

Advisories explain the risk output; they never change the risk score.
"""

LEVELS = ["low", "moderate", "high", "very_high", "extreme"]

GENERAL = {
    "low": "Current heat risk is low. Maintain normal hydration and routine heat-safety precautions.",
    "moderate": "Moderate heat risk. Stay hydrated and take sensible precautions during prolonged outdoor activity.",
    "high": "Elevated heat risk. Stay hydrated, limit strenuous outdoor activity and take regular breaks in shade or cool areas.",
    "very_high": "High heat risk. Reduce prolonged outdoor activity, stay hydrated, seek shade or cooling and monitor local heat-health guidance.",
    "extreme": "Extreme heat conditions. Avoid outdoor exposure in the afternoon, stay hydrated, use cooling measures and follow official heat-health guidance.",
}

AUDIENCES = {
    "outdoor_workers": {
        "low": "Normal work routines are fine; keep drinking water available.",
        "moderate": "Drink water every 20 minutes even if not thirsty, and wear light, loose clothing and a head covering.",
        "high": "Schedule heavy work for early morning or evening, take shaded rest breaks every hour and work in pairs.",
        "very_high": "Avoid heavy work between 12:00 and 16:00, rest in shade at least 10 minutes every hour, and know the signs of heat exhaustion.",
        "extreme": "Stop strenuous outdoor work between 11:00 and 17:00. Keep oral rehydration solution on hand and seek medical help for confusion, fainting or no sweating.",
    },
    "elderly": {
        "low": "No special precautions needed beyond regular fluids.",
        "moderate": "Keep drinking fluids through the day and stay in well-ventilated rooms in the afternoon.",
        "high": "Stay indoors during the hottest hours, use fans or cooling, and have someone check on you daily.",
        "very_high": "Stay in the coolest room available, avoid going out between 12:00 and 16:00, and keep medicines away from heat. Ask a neighbour or relative to check on you twice a day.",
        "extreme": "Do not go out in the afternoon. Use cool showers or wet cloths to cool down, drink fluids regularly and seek medical help quickly for dizziness, confusion or breathlessness.",
    },
    "children": {
        "low": "Normal play is fine; send a water bottle to school.",
        "moderate": "Make sure children drink water regularly and wear hats outdoors.",
        "high": "Limit outdoor play to mornings and evenings and never leave children in parked vehicles.",
        "very_high": "Keep children indoors between 12:00 and 16:00, dress them in light cotton and watch for irritability, vomiting or unusual sleepiness.",
        "extreme": "Keep children and infants indoors and cool during the day, give fluids often, and seek medical help immediately for signs of heat illness.",
    },
}


def level_key(risk_category):
    """'Very High' -> 'very_high'."""
    return str(risk_category).strip().lower().replace(" ", "_")


def generate_advisory(risk_category, severity, predicted_tmax, heatwave_probability, humidity=None):
    """Return {headline, general, audiences: {...}, context: [...]}."""
    level = level_key(risk_category)
    if level not in GENERAL:
        level = "low"
    severity_text = str(severity).lower()
    # A model- or rule-indicated severe heatwave never gets a mild advisory.
    if "severe" in severity_text:
        level = "extreme"
    elif severity_text == "heatwave" and LEVELS.index(level) < LEVELS.index("very_high"):
        level = "very_high"

    context = []
    if predicted_tmax is not None and predicted_tmax >= 40:
        context.append(f"Forecast maximum temperature is {predicted_tmax:.1f}°C.")
    if heatwave_probability is not None and heatwave_probability >= 0.05:
        context.append(f"Estimated heatwave probability is {heatwave_probability * 100:.0f}%.")
    if humidity is not None and humidity >= 70 and predicted_tmax is not None and predicted_tmax >= 33:
        context.append("High humidity will make the heat feel worse.")

    return {
        "level": level,
        "general": GENERAL[level],
        "audiences": {name: text[level] for name, text in AUDIENCES.items()},
        "context": context,
    }
