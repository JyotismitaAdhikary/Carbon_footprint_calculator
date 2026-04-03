"""
Carbon Footprint Calculator - Core Logic
Handles calculations, predictions, and recommendations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class EmissionFactors:
    """
    Emission factors in kg CO2 per unit.
    Sources: EPA, DEFRA, IPCC guidelines
    """
    
    # Transportation (per km)
    CAR_PETROL: float = 0.21
    CAR_DIESEL: float = 0.17
    CAR_HYBRID: float = 0.12
    CAR_ELECTRIC: float = 0.05
    PUBLIC_TRANSPORT: float = 0.089
    FLIGHT_DOMESTIC_PER_KM: float = 0.255
    FLIGHT_SHORT_HAUL_PER_KM: float = 0.156
    FLIGHT_LONG_HAUL_PER_KM: float = 0.195
    
    # Average flight distances (km)
    FLIGHT_DOMESTIC_DISTANCE: int = 800
    FLIGHT_SHORT_HAUL_DISTANCE: int = 1500
    FLIGHT_LONG_HAUL_DISTANCE: int = 8000
    
    # Home energy
    ELECTRICITY_KWH: float = 0.42  # Global average
    NATURAL_GAS_M3: float = 2.0
    HEATING_OIL_LITER: float = 2.68
    
    # Diet (kg CO2 per day)
    DIET_MEAT_HEAVY: float = 7.2
    DIET_MEAT_MEDIUM: float = 5.6
    DIET_MEAT_LOW: float = 4.1
    DIET_PESCATARIAN: float = 3.9
    DIET_VEGETARIAN: float = 3.8
    DIET_VEGAN: float = 2.9
    
    # Consumption (kg CO2 per year)
    SHOPPING_HIGH: float = 1200
    SHOPPING_MEDIUM: float = 600
    SHOPPING_LOW: float = 300
    SHOPPING_MINIMAL: float = 150


class CarbonCalculator:
    """Calculates carbon footprint from user inputs."""
    
    def __init__(self):
        self.factors = EmissionFactors()
    
    def calculate(self, user_data: Dict) -> Dict:
        """
        Calculate annual carbon footprint.
        
        Args:
            user_data: Dictionary with user lifestyle inputs
            
        Returns:
            Dictionary with total and category breakdown
        """
        breakdown = {}
        
        # === Transportation ===
        transport = 0
        
        # Car emissions
        car_type = user_data.get('car_type', 'petrol')
        car_km_weekly = user_data.get('car_km_weekly', 0)
        
        car_factors = {
            'petrol': self.factors.CAR_PETROL,
            'diesel': self.factors.CAR_DIESEL,
            'hybrid': self.factors.CAR_HYBRID,
            'electric': self.factors.CAR_ELECTRIC,
            'none': 0
        }
        transport += car_km_weekly * 52 * car_factors.get(car_type, 0)
        
        # Public transport
        public_km_weekly = user_data.get('public_transport_km_weekly', 0)
        transport += public_km_weekly * 52 * self.factors.PUBLIC_TRANSPORT
        
        # Flights
        flights_domestic = user_data.get('flights_domestic', 0)
        flights_short = user_data.get('flights_short_haul', 0)
        flights_long = user_data.get('flights_long_haul', 0)
        
        transport += (flights_domestic * self.factors.FLIGHT_DOMESTIC_DISTANCE * 
                      self.factors.FLIGHT_DOMESTIC_PER_KM)
        transport += (flights_short * self.factors.FLIGHT_SHORT_HAUL_DISTANCE * 
                      self.factors.FLIGHT_SHORT_HAUL_PER_KM)
        transport += (flights_long * self.factors.FLIGHT_LONG_HAUL_DISTANCE * 
                      self.factors.FLIGHT_LONG_HAUL_PER_KM)
        
        breakdown['Transportation'] = round(transport, 2)
        
        # === Home Energy ===
        home = 0
        
        electricity_kwh = user_data.get('electricity_kwh_monthly', 0)
        home += electricity_kwh * 12 * self.factors.ELECTRICITY_KWH
        
        gas_m3 = user_data.get('gas_m3_monthly', 0)
        home += gas_m3 * 12 * self.factors.NATURAL_GAS_M3
        
        # Renewable energy offset
        renewable_percent = user_data.get('renewable_percent', 0)
        home *= (1 - renewable_percent / 100)
        
        breakdown['Home Energy'] = round(home, 2)
        
        # === Diet ===
        diet_type = user_data.get('diet_type', 'meat_medium')
        diet_factors = {
            'meat_heavy': self.factors.DIET_MEAT_HEAVY,
            'meat_medium': self.factors.DIET_MEAT_MEDIUM,
            'meat_low': self.factors.DIET_MEAT_LOW,
            'pescatarian': self.factors.DIET_PESCATARIAN,
            'vegetarian': self.factors.DIET_VEGETARIAN,
            'vegan': self.factors.DIET_VEGAN
        }
        diet = diet_factors.get(diet_type, self.factors.DIET_MEAT_MEDIUM) * 365
        
        # Local food bonus
        local_food_percent = user_data.get('local_food_percent', 0)
        diet *= (1 - local_food_percent / 100 * 0.1)  # Up to 10% reduction
        
        breakdown['Diet'] = round(diet, 2)
        
        # === Consumption ===
        shopping_level = user_data.get('shopping_level', 'medium')
        shopping_factors = {
            'high': self.factors.SHOPPING_HIGH,
            'medium': self.factors.SHOPPING_MEDIUM,
            'low': self.factors.SHOPPING_LOW,
            'minimal': self.factors.SHOPPING_MINIMAL
        }
        consumption = shopping_factors.get(shopping_level, self.factors.SHOPPING_MEDIUM)
        
        breakdown['Consumption'] = round(consumption, 2)
        
        # === Total ===
        total_kg = sum(breakdown.values())
        
        return {
            'total_kg': round(total_kg, 2),
            'total_tons': round(total_kg / 1000, 2),
            'breakdown': breakdown,
            'breakdown_percent': {
                k: round(v / total_kg * 100, 1) 
                for k, v in breakdown.items()
            }
        }


class CarbonPredictor:
    """Predicts future carbon footprint trends."""
    
    @staticmethod
    def predict(
        current_footprint: float,
        years: int = 10,
        growth_rate: float = 0.02,
        reduction_actions: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Predict future emissions.
        
        Args:
            current_footprint: Current annual footprint in kg CO2
            years: Years to project
            growth_rate: Annual growth rate (default 2%)
            reduction_actions: List of reduction actions with start year and impact
            
        Returns:
            Prediction data for visualization
        """
        baseline = []
        with_actions = []
        cumulative_baseline = 0
        cumulative_actions = 0
        
        for year in range(years + 1):
            # Baseline projection
            yearly_baseline = current_footprint * ((1 + growth_rate) ** year)
            cumulative_baseline += yearly_baseline
            
            # With reduction actions
            yearly_reduced = yearly_baseline
            if reduction_actions:
                for action in reduction_actions:
                    if year >= action.get('start_year', 0):
                        yearly_reduced -= action.get('annual_savings_kg', 0)
            
            yearly_reduced = max(0, yearly_reduced)
            cumulative_actions += yearly_reduced
            
            baseline.append({
                'year': year,
                'yearly_tons': round(yearly_baseline / 1000, 2),
                'cumulative_tons': round(cumulative_baseline / 1000, 2)
            })
            
            with_actions.append({
                'year': year,
                'yearly_tons': round(yearly_reduced / 1000, 2),
                'cumulative_tons': round(cumulative_actions / 1000, 2)
            })
        
        return {
            'baseline': baseline,
            'with_actions': with_actions,
            'total_baseline_tons': round(cumulative_baseline / 1000, 2),
            'total_with_actions_tons': round(cumulative_actions / 1000, 2),
            'total_savings_tons': round((cumulative_baseline - cumulative_actions) / 1000, 2)
        }


class RecommendationEngine:
    """Generates personalized recommendations."""
    
    def __init__(self):
        self.calculator = CarbonCalculator()
    
    def get_recommendations(self, user_data: Dict) -> List[Dict]:
        """
        Generate recommendations based on user data.
        
        Returns list of recommendations sorted by impact.
        """
        recommendations = []
        current = self.calculator.calculate(user_data)
        
        # Transportation recommendations
        if user_data.get('car_type') in ['petrol', 'diesel']:
            # Electric vehicle
            modified = {**user_data, 'car_type': 'electric'}
            new_calc = self.calculator.calculate(modified)
            savings = current['total_kg'] - new_calc['total_kg']
            if savings > 100:
                recommendations.append({
                    'category': '🚗 Transportation',
                    'action': 'Switch to an electric vehicle',
                    'description': 'EVs produce significantly fewer emissions, especially with renewable charging.',
                    'savings_kg': round(savings, 0),
                    'savings_percent': round(savings / current['total_kg'] * 100, 1),
                    'difficulty': 'High',
                    'cost': 'High upfront, lower running costs'
                })
            
            # Hybrid option
            modified = {**user_data, 'car_type': 'hybrid'}
            new_calc = self.calculator.calculate(modified)
            savings = current['total_kg'] - new_calc['total_kg']
            if savings > 50:
                recommendations.append({
                    'category': '🚗 Transportation',
                    'action': 'Switch to a hybrid vehicle',
                    'description': 'Hybrids offer a middle ground with good fuel efficiency.',
                    'savings_kg': round(savings, 0),
                    'savings_percent': round(savings / current['total_kg'] * 100, 1),
                    'difficulty': 'Medium',
                    'cost': 'Medium'
                })
        
        # Reduce driving
        if user_data.get('car_km_weekly', 0) > 50:
            modified = {**user_data, 'car_km_weekly': user_data['car_km_weekly'] * 0.5}
            new_calc = self.calculator.calculate(modified)
            savings = current['total_kg'] - new_calc['total_kg']
            recommendations.append({
                'category': '🚗 Transportation',
                'action': 'Reduce car travel by 50%',
                'description': 'Work from home, carpool, or combine trips.',
                'savings_kg': round(savings, 0),
                'savings_percent': round(savings / current['total_kg'] * 100, 1),
                'difficulty': 'Medium',
                'cost': 'Saves money'
            })
        
        # Flight reduction
        total_flights = (user_data.get('flights_domestic', 0) + 
                        user_data.get('flights_short_haul', 0) + 
                        user_data.get('flights_long_haul', 0))
        
        if user_data.get('flights_long_haul', 0) >= 2:
            modified = {**user_data, 'flights_long_haul': 1}
            new_calc = self.calculator.calculate(modified)
            savings = current['total_kg'] - new_calc['total_kg']
            recommendations.append({
                'category': '✈️ Travel',
                'action': 'Take one fewer long-haul flight per year',
                'description': 'One transatlantic flight can equal a year of driving.',
                'savings_kg': round(savings, 0),
                'savings_percent': round(savings / current['total_kg'] * 100, 1),
                'difficulty': 'Medium',
                'cost': 'Saves money'
            })
        
        # Diet recommendations
        diet = user_data.get('diet_type', 'meat_medium')
        if diet in ['meat_heavy', 'meat_medium']:
            modified = {**user_data, 'diet_type': 'vegetarian'}
            new_calc = self.calculator.calculate(modified)
            savings = current['total_kg'] - new_calc['total_kg']
            recommendations.append({
                'category': '🥗 Diet',
                'action': 'Adopt a vegetarian diet',
                'description': 'Plant-based diets have significantly lower carbon footprints.',
                'savings_kg': round(savings, 0),
                'savings_percent': round(savings / current['total_kg'] * 100, 1),
                'difficulty': 'Medium',
                'cost': 'Often cheaper'
            })
            
            # Meat reduction
            if diet == 'meat_heavy':
                modified = {**user_data, 'diet_type': 'meat_low'}
                new_calc = self.calculator.calculate(modified)
                savings = current['total_kg'] - new_calc['total_kg']
                recommendations.append({
                    'category': '🥗 Diet',
                    'action': 'Reduce meat to once per week',
                    'description': 'Even small reductions in meat consumption help significantly.',
                    'savings_kg': round(savings, 0),
                    'savings_percent': round(savings / current['total_kg'] * 100, 1),
                    'difficulty': 'Low',
                    'cost': 'Saves money'
                })
        
        # Energy recommendations
        if user_data.get('renewable_percent', 0) < 50:
            modified = {**user_data, 'renewable_percent': 100}
            new_calc = self.calculator.calculate(modified)
            savings = current['total_kg'] - new_calc['total_kg']
            if savings > 100:
                recommendations.append({
                    'category': '⚡ Energy',
                    'action': 'Switch to 100% renewable electricity',
                    'description': 'Many providers offer green energy tariffs.',
                    'savings_kg': round(savings, 0),
                    'savings_percent': round(savings / current['total_kg'] * 100, 1),
                    'difficulty': 'Low',
                    'cost': 'Similar or slightly higher'
                })
        
        if user_data.get('electricity_kwh_monthly', 0) > 200:
            modified = {**user_data, 'electricity_kwh_monthly': user_data['electricity_kwh_monthly'] * 0.7}
            new_calc = self.calculator.calculate(modified)
            savings = current['total_kg'] - new_calc['total_kg']
            recommendations.append({
                'category': '⚡ Energy',
                'action': 'Reduce electricity usage by 30%',
                'description': 'LED bulbs, efficient appliances, smart power strips.',
                'savings_kg': round(savings, 0),
                'savings_percent': round(savings / current['total_kg'] * 100, 1),
                'difficulty': 'Low',
                'cost': 'Saves money'
            })
        
        # Shopping recommendations
        if user_data.get('shopping_level') in ['high', 'medium']:
            target = 'low' if user_data.get('shopping_level') == 'high' else 'minimal'
            modified = {**user_data, 'shopping_level': target}
            new_calc = self.calculator.calculate(modified)
            savings = current['total_kg'] - new_calc['total_kg']
            recommendations.append({
                'category': '🛒 Consumption',
                'action': 'Reduce consumption and buy secondhand',
                'description': 'Extend product life, repair, and choose quality over quantity.',
                'savings_kg': round(savings, 0),
                'savings_percent': round(savings / current['total_kg'] * 100, 1),
                'difficulty': 'Low',
                'cost': 'Saves money'
            })
        
        # Sort by savings impact
        recommendations.sort(key=lambda x: x['savings_kg'], reverse=True)
        
        return recommendations


# Global average comparisons (tons CO2/year)
GLOBAL_BENCHMARKS = {
    'World Average': 4.7,
    'USA': 15.5,
    'EU Average': 7.0,
    'China': 7.4,
    'India': 1.9,
    'UK': 5.5,
    'Germany': 8.1,
    'France': 4.6,
    'Japan': 8.7,
    'Brazil': 2.2,
    'Paris Agreement Target (2030)': 2.0,
    'Net Zero Target': 0.0
}

