"""
Carbon Footprint Chatbot - LLM-powered assistant
Uses Claude API with your emission factors as knowledge base.
"""

import json
import anthropic
import streamlit as st
def _get_client():
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
# Your existing emission factors as knowledge base for the LLM
CARBON_KNOWLEDGE_BASE = """
You are a friendly and knowledgeable carbon footprint assistant built into a personal carbon calculator app.
You have access to the user's calculated carbon footprint data and can answer questions about it.

EMISSION FACTORS YOU KNOW:
- Petrol car: 0.21 kg CO2/km, Diesel: 0.17, Hybrid: 0.12, Electric: 0.05
- Public transport: 0.089 kg CO2/km
- Domestic flight: 0.255 kg CO2/km (~800km avg), Short-haul: 0.156 kg CO2/km (~1500km), Long-haul: 0.195 kg CO2/km (~8000km)
- Electricity: 0.42 kg CO2/kWh (global average)
- Natural gas: 2.0 kg CO2/m³
- Diet (per day): Meat-heavy 7.2kg, Meat-medium 5.6kg, Meat-low 4.1kg, Pescatarian 3.9kg, Vegetarian 3.8kg, Vegan 2.9kg
- Shopping: High 1200kg/yr, Medium 600kg/yr, Low 300kg/yr, Minimal 150kg/yr

GLOBAL BENCHMARKS (tons CO2/year):
- World Average: 4.7, USA: 15.5, EU: 7.0, UK: 5.5, India: 1.9
- Paris Agreement 2030 Target: 2.0 tons/year

YOUR ROLE:
- Help users understand their personal carbon footprint results
- Explain WHY certain activities have high/low emissions
- Suggest practical, realistic changes based on their actual data
- Be encouraging, not preachy — small changes matter
- Give specific numbers when possible (e.g. "switching to electric would save you X tons")
- Keep responses concise (3-5 sentences max) unless asked for detail
- If asked something outside carbon/environment, gently redirect back to the topic

TONE: Warm, encouraging, data-driven. Like a knowledgeable friend, not a lecture.
"""


def build_system_prompt(user_data: dict, results: dict) -> str:
    """
    Inject the user's actual footprint data into the system prompt
    so the LLM can give personalised answers.
    """
    if not user_data or not results:
        return CARBON_KNOWLEDGE_BASE + "\nNo user data available yet — ask them to complete the calculator first."

    user_context = f"""
CURRENT USER'S DATA:
- Car type: {user_data.get('car_type', 'unknown')}, driving {user_data.get('car_km_weekly', 0)} km/week
- Public transport: {user_data.get('public_transport_km_weekly', 0)} km/week
- Flights per year: {user_data.get('flights_domestic', 0)} domestic, {user_data.get('flights_short_haul', 0)} short-haul, {user_data.get('flights_long_haul', 0)} long-haul
- Electricity: {user_data.get('electricity_kwh_monthly', 0)} kWh/month, {user_data.get('renewable_percent', 0)}% renewable
- Gas: {user_data.get('gas_m3_monthly', 0)} m³/month
- Diet: {user_data.get('diet_type', 'unknown')}
- Local food: {user_data.get('local_food_percent', 0)}%
- Shopping level: {user_data.get('shopping_level', 'unknown')}

THEIR RESULTS:
- Total annual footprint: {results.get('total_tons', 0)} tons CO2
- Breakdown: {json.dumps(results.get('breakdown', {}), indent=2)}
- Biggest category: {max(results.get('breakdown', {}).items(), key=lambda x: x[1])[0] if results.get('breakdown') else 'unknown'}

When answering, refer to their specific numbers. For example if they ask 
"what's my biggest impact?" use their actual breakdown data above.
"""
    return CARBON_KNOWLEDGE_BASE + user_context


def get_chat_response(
    messages: list,
    user_data: dict,
    results: dict
) -> str:
    """
    Send conversation history to Claude and get a response.
    
    Args:
        messages: List of {"role": "user"/"assistant", "content": "..."} dicts
        user_data: User's lifestyle inputs from session state
        results: Calculated footprint results from session state
    
    Returns:
        Assistant's response string
    """
    client = _get_client()

    system_prompt = build_system_prompt(user_data, results)

    import time
    for attempt in range(3):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                system=system_prompt,
                messages=messages
            )
            return response.content[0].text
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < 2:
                time.sleep(3)
                continue
            raise


# Suggested starter questions to show in the UI
SUGGESTED_QUESTIONS = [
    "What's my biggest source of emissions?",
    "What's the single most impactful change I can make?",
    "How do I compare to someone in the UK?",
    "Why is flying so bad for the environment?",
    "How many years until I hit the Paris target if I switch to electric?",
    "What does my diet change actually save in real terms?",
]
