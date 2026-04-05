"""
Carbon Footprint Calculator - Streamlit App
Calculate, predict, and reduce your environmental impact.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from carbon_model import (
    CarbonCalculator, 
    CarbonPredictor, 
    RecommendationEngine,
    GLOBAL_BENCHMARKS
)
from ml_model import ml_predict_trajectory, ACTION_TO_FEATURE_CHANGE
from chatbot import get_chat_response, SUGGESTED_QUESTIONS
# Page configuration
st.set_page_config(
    page_title="Carbon Footprint Calculator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #2ecc71, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2ecc71;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
    }
    
    .recommendation-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #2ecc71;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'calculated' not in st.session_state:
        st.session_state.calculated = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'selected_actions' not in st.session_state:
        st.session_state.selected_actions = []


def render_header():
    """Render app header."""
    st.markdown('<h1 class="main-header">🌍 Carbon Footprint Calculator</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #94a3b8;">Calculate your environmental impact and discover ways to reduce it</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")


def render_calculator_tab():
    """Render the calculator input form."""
    
    st.subheader("📝 Enter Your Lifestyle Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚗 Transportation")
        
        car_type = st.selectbox(
            "What type of vehicle do you drive?",
            options=['petrol', 'diesel', 'hybrid', 'electric', 'none'],
            format_func=lambda x: {
                'petrol': '⛽ Petrol/Gasoline',
                'diesel': '⛽ Diesel',
                'hybrid': '🔋 Hybrid',
                'electric': '⚡ Electric',
                'none': '🚫 No car'
            }[x],
            index=0
        )
        
        car_km_weekly = st.slider(
            "Weekly driving distance (km)",
            min_value=0,
            max_value=500,
            value=100,
            step=10,
            disabled=(car_type == 'none')
        )
        
        public_transport_km = st.slider(
            "Weekly public transport distance (km)",
            min_value=0,
            max_value=300,
            value=30,
            step=5
        )
        
        st.markdown("#### ✈️ Annual Flights")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            flights_domestic = st.number_input(
                "Domestic",
                min_value=0,
                max_value=50,
                value=2,
                help="Flights within your country"
            )
        with col_f2:
            flights_short = st.number_input(
                "Short-haul",
                min_value=0,
                max_value=30,
                value=1,
                help="Flights < 3 hours"
            )
        with col_f3:
            flights_long = st.number_input(
                "Long-haul",
                min_value=0,
                max_value=20,
                value=1,
                help="Flights > 3 hours"
            )
    
    with col2:
        st.markdown("### 🏠 Home Energy")
        
        electricity_kwh = st.slider(
            "Monthly electricity usage (kWh)",
            min_value=50,
            max_value=1000,
            value=300,
            step=10,
            help="Check your electricity bill"
        )
        
        gas_m3 = st.slider(
            "Monthly natural gas (m³)",
            min_value=0,
            max_value=200,
            value=40,
            step=5,
            help="Set to 0 if you don't use gas"
        )
        
        renewable_percent = st.slider(
            "Renewable energy percentage",
            min_value=0,
            max_value=100,
            value=0,
            step=10,
            help="If you have solar panels or green tariff"
        )
        
        st.markdown("### 🍽️ Diet & Lifestyle")
        
        diet_type = st.selectbox(
            "What best describes your diet?",
            options=['meat_heavy', 'meat_medium', 'meat_low', 'pescatarian', 'vegetarian', 'vegan'],
            format_func=lambda x: {
                'meat_heavy': '🥩 Meat with most meals',
                'meat_medium': '🍖 Meat a few times per week',
                'meat_low': '🍗 Meat once a week or less',
                'pescatarian': '🐟 Pescatarian (fish, no meat)',
                'vegetarian': '🥬 Vegetarian',
                'vegan': '🌱 Vegan'
            }[x],
            index=1
        )
        
        local_food = st.slider(
            "Local/seasonal food percentage",
            min_value=0,
            max_value=100,
            value=20,
            step=10
        )
        
        shopping_level = st.selectbox(
            "Shopping & consumption level",
            options=['high', 'medium', 'low', 'minimal'],
            format_func=lambda x: {
                'high': '🛍️ High (frequent new purchases)',
                'medium': '🛒 Average',
                'low': '♻️ Low (mostly secondhand/repair)',
                'minimal': '🌿 Minimal (essentials only)'
            }[x],
            index=1
        )
    
    st.markdown("---")
    
    # Calculate button
    if st.button("🌱 Calculate My Carbon Footprint", type="primary", use_container_width=True):
        user_data = {
            'car_type': car_type,
            'car_km_weekly': car_km_weekly if car_type != 'none' else 0,
            'public_transport_km_weekly': public_transport_km,
            'flights_domestic': flights_domestic,
            'flights_short_haul': flights_short,
            'flights_long_haul': flights_long,
            'electricity_kwh_monthly': electricity_kwh,
            'gas_m3_monthly': gas_m3,
            'renewable_percent': renewable_percent,
            'diet_type': diet_type,
            'local_food_percent': local_food,
            'shopping_level': shopping_level
        }
        
        calculator = CarbonCalculator()
        results = calculator.calculate(user_data)
        
        st.session_state.user_data = user_data
        st.session_state.results = results
        st.session_state.calculated = True
        
        st.success("✅ Calculation complete! Check the Results tab.")
        st.rerun()


def render_results_tab():
    """Render results and predictions."""
    
    if not st.session_state.calculated:
        st.info("👆 Please complete the calculator first to see your results.")
        return
    
    results = st.session_state.results
    user_data = st.session_state.user_data
    
    # Summary metrics
    st.subheader("📊 Your Carbon Footprint Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Annual Footprint",
            value=f"{results['total_tons']:.1f} tons",
            help="Total CO₂ equivalent per year"
        )
    
    with col2:
        global_avg = GLOBAL_BENCHMARKS['World Average']
        diff = results['total_tons'] - global_avg
        st.metric(
            label="vs World Average",
            value=f"{diff:+.1f} tons",
            delta=f"{(diff/global_avg)*100:+.0f}%",
            delta_color="inverse"
        )
    
    with col3:
        target = GLOBAL_BENCHMARKS['Paris Agreement Target (2030)']
        reduction_needed = max(0, results['total_tons'] - target)
        st.metric(
            label="Reduction Needed",
            value=f"{reduction_needed:.1f} tons",
            help="To meet Paris Agreement targets"
        )
    
    with col4:
        trees_needed = int(results['total_kg'] / 21)  # ~21kg CO2 per tree per year
        st.metric(
            label="Trees to Offset",
            value=f"{trees_needed:,}",
            help="Trees needed to absorb your annual emissions"
        )
    
    st.markdown("---")
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### 📈 Breakdown by Category")
        
        breakdown_df = pd.DataFrame({
            'Category': list(results['breakdown'].keys()),
            'Emissions (kg CO₂)': list(results['breakdown'].values()),
            'Percentage': list(results['breakdown_percent'].values())
        })
        
        fig_pie = px.pie(
            breakdown_df,
            values='Emissions (kg CO₂)',
            names='Category',
            color_discrete_sequence=px.colors.sequential.Viridis,
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_chart2:
        st.markdown("#### 🌍 How You Compare")
        
        comparison_data = [
            {'Region': region, 'Emissions (tons)': value, 'Type': 'Benchmark'}
            for region, value in list(GLOBAL_BENCHMARKS.items())[:8]
        ]
        comparison_data.insert(0, {
            'Region': '👤 You',
            'Emissions (tons)': results['total_tons'],
            'Type': 'Your Footprint'
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig_bar = px.bar(
            comparison_df,
            x='Emissions (tons)',
            y='Region',
            orientation='h',
            color='Type',
            color_discrete_map={
                'Your Footprint': '#e74c3c',
                'Benchmark': '#3498db'
            }
        )
        fig_bar.update_layout(
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(t=40, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # 10-Year Projection
    st.subheader("🔮 10-Year Projection")
    
    col_proj1, col_proj2 = st.columns([3, 1])
    
    with col_proj2:
        growth_rate = st.slider(
            "Annual lifestyle growth rate",
            min_value=-5,
            max_value=10,
            value=2,
            format="%d%%",
            help="How much your consumption might increase yearly"
        ) / 100
    
    # Convert selected recommendations into ML feature-level changes
    planned_changes = []
    for action in st.session_state.get('selected_actions', []):
        mapping = ACTION_TO_FEATURE_CHANGE.get(action['action'])
        if mapping:
            change = {'year': 1, 'field': mapping['field'], 'value': mapping['value']}

            # Handle relative values
            if mapping['value'] == '__half__':
                change['value'] = user_data.get('car_km_weekly', 0) * 0.5
            elif mapping['value'] == '__70pct__':
                change['value'] = user_data.get('electricity_kwh_monthly', 0) * 0.7
            elif mapping['value'] == '__minus1__':
                change['value'] = max(0, user_data.get('flights_long_haul', 0) - 1)

            planned_changes.append(change)

    with st.spinner("🤖 Running ML model prediction..."):
        predictions = ml_predict_trajectory(
            user_data=user_data,
            planned_changes=planned_changes,
            years=10,
            growth_rate=growth_rate
        )
    
    with col_proj1:
        # Create projection chart
        baseline_df = pd.DataFrame(predictions['baseline'])
        baseline_df['Scenario'] = 'Business as Usual'
        
        actions_df = pd.DataFrame(predictions['with_actions'])
        actions_df['Scenario'] = 'With Reductions'
        
        projection_df = pd.concat([baseline_df, actions_df])
        
        fig_proj = px.line(
            projection_df,
            x='year',
            y='yearly_tons',
            color='Scenario',
            markers=True,
            labels={'year': 'Year', 'yearly_tons': 'Annual Emissions (tons CO₂)'},
            color_discrete_map={
                'Business as Usual': '#e74c3c',
                'With Reductions': '#2ecc71'
            }
        )
        
        # Add Paris target line
        fig_proj.add_hline(
            y=2.0,
            line_dash="dash",
            line_color="#f39c12",
            annotation_text="Paris 2030 Target",
            annotation_position="bottom right"
        )
        
        fig_proj.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        st.plotly_chart(fig_proj, use_container_width=True)
    
    # Projection summary
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.metric(
            label="10-Year Total (Baseline)",
            value=f"{predictions['total_baseline_tons']:.1f} tons"
        )
    
    with col_s2:
        st.metric(
            label="10-Year Total (With Actions)",
            value=f"{predictions['total_with_actions_tons']:.1f} tons"
        )
    
    with col_s3:
        st.metric(
            label="Potential Savings",
            value=f"{predictions['total_savings_tons']:.1f} tons",
            delta=f"-{(predictions['total_savings_tons']/predictions['total_baseline_tons'])*100:.0f}%"
        )


def render_tips_tab():
    """Render recommendations and tips."""
    
    if not st.session_state.calculated:
        st.info("👆 Please complete the calculator first to see personalized tips.")
        return
    
    user_data = st.session_state.user_data
    results = st.session_state.results
    
    st.subheader("💡 Personalized Recommendations")
    st.markdown("Select actions to see their combined impact on your 10-year projection.")
    
    engine = RecommendationEngine()
    recommendations = engine.get_recommendations(user_data)
    
    if not recommendations:
        st.success("🎉 Great job! Your carbon footprint is already quite low. Keep up the good work!")
        return
    
    # Difficulty filter
    difficulty_filter = st.multiselect(
        "Filter by difficulty",
        options=['Low', 'Medium', 'High'],
        default=['Low', 'Medium', 'High']
    )
    
    filtered_recs = [r for r in recommendations if r['difficulty'] in difficulty_filter]
    
    # Track selected actions
    selected_actions = []
    
    for i, rec in enumerate(filtered_recs):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected = st.checkbox(
                f"**{rec['action']}**",
                key=f"rec_{i}",
                help=rec['description']
            )
            
            st.markdown(f"""
            <div style="margin-left: 28px; margin-top: -10px; margin-bottom: 15px;">
                <span style="color: #94a3b8; font-size: 0.9rem;">{rec['description']}</span><br>
                <span style="background: rgba(46, 204, 113, 0.2); padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;">
                    {rec['category']}
                </span>
                <span style="background: rgba(52, 152, 219, 0.2); padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; margin-left: 5px;">
                    {rec['difficulty']} effort
                </span>
                <span style="background: rgba(241, 196, 15, 0.2); padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; margin-left: 5px;">
                    {rec['cost']}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            if selected:
                selected_actions.append(rec)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #2ecc71;">
                    -{rec['savings_kg']/1000:.1f}t
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8;">
                    {rec['savings_percent']}% reduction
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Update session state
    st.session_state.selected_actions = selected_actions
    
    # Impact summary
    if selected_actions:
        st.markdown("---")
        st.subheader("📊 Combined Impact of Selected Actions")
        
        total_savings_kg = sum(a['savings_kg'] for a in selected_actions)
        total_savings_percent = (total_savings_kg / results['total_kg']) * 100
        new_footprint = results['total_tons'] - (total_savings_kg / 1000)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Current Footprint",
                value=f"{results['total_tons']:.1f} tons/year"
            )
        
        with col2:
            st.metric(
                label="New Footprint",
                value=f"{new_footprint:.1f} tons/year",
                delta=f"-{total_savings_percent:.0f}%"
            )
        
        with col3:
            target = GLOBAL_BENCHMARKS['Paris Agreement Target (2030)']
            if new_footprint <= target:
                st.success(f"✅ You'd meet the Paris target!")
            else:
                st.warning(f"⚠️ Still {new_footprint - target:.1f}t above target")
        
        # Visual comparison
        comparison_data = pd.DataFrame({
            'Scenario': ['Current', 'With Actions', 'Paris Target'],
            'Emissions': [results['total_tons'], new_footprint, 2.0]
        })
        
        fig = px.bar(
            comparison_data,
            x='Scenario',
            y='Emissions',
            color='Scenario',
            color_discrete_map={
                'Current': '#e74c3c',
                'With Actions': '#2ecc71',
                'Paris Target': '#3498db'
            }
        )
        fig.update_layout(
            showlegend=False,
            yaxis_title="Tons CO₂/year",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 Go back to the **Results** tab to see how these actions affect your 10-year projection!")
    
    # Additional quick tips
    st.markdown("---")
    st.subheader("⚡ Quick Wins (Easy Changes)")
    
    quick_tips = [
        ("🔌", "Unplug devices when not in use", "~100 kg CO₂/year"),
        ("🌡️", "Lower heating by 1°C", "~300 kg CO₂/year"),
        ("🚿", "Take shorter showers", "~200 kg CO₂/year"),
        ("🛒", "Buy local and seasonal produce", "~100-200 kg CO₂/year"),
        ("💡", "Switch to LED bulbs throughout home", "~50 kg CO₂/year"),
        ("🧺", "Wash clothes at 30°C instead of 40°C", "~60 kg CO₂/year"),
        ("🚰", "Fix dripping taps", "~30 kg CO₂/year"),
        ("📦", "Reduce food waste by 50%", "~200 kg CO₂/year"),
    ]
    
    cols = st.columns(2)
    for i, (icon, tip, savings) in enumerate(quick_tips):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                <span style="font-size: 1.5rem;">{icon}</span>
                <strong>{tip}</strong><br>
                <span style="color: #2ecc71; font-size: 0.9rem;">Saves {savings}</span>
            </div>
            """, unsafe_allow_html=True)


def render_about_tab():
    """Render about and methodology information."""
    
    st.subheader("ℹ️ About This Calculator")
    
    st.markdown("""
    ### How It Works
    
    This calculator estimates your annual carbon footprint based on key lifestyle factors:
    
    - **Transportation**: Vehicle type, distance driven, public transport usage, and flights
    - **Home Energy**: Electricity and gas consumption, renewable energy usage
    - **Diet**: Food choices from vegan to meat-heavy diets
    - **Consumption**: Shopping habits and material consumption
    
    ### Emission Factors
    
    The calculations use emission factors from trusted sources including:
    - EPA (Environmental Protection Agency)
    - DEFRA (UK Department for Environment, Food & Rural Affairs)
    - IPCC (Intergovernmental Panel on Climate Change)
    
    ### Limitations
    
    This calculator provides estimates. Actual emissions vary based on:
    - Your country's electricity grid mix
    - Specific vehicle efficiency
    - Local food production methods
    - Building efficiency and heating systems
    
    ### Data Privacy
    
    All calculations happen in your browser. No personal data is stored or transmitted.

    ### Creator
    This application was developed by Jyotismita Adhikary, with some support from Claude in designing the web interface.

    ### Updates
    Working on optimising chatbot! Go green!
    
    """)
    
    st.markdown("---")
    
    st.subheader("📊 Emission Factors Used")
    
    factors_data = {
        'Category': [
            'Petrol Car', 'Diesel Car', 'Hybrid Car', 'Electric Car',
            'Public Transport', 'Domestic Flight', 'Long-haul Flight',
            'Electricity', 'Natural Gas',
            'Meat-heavy Diet', 'Vegetarian Diet', 'Vegan Diet'
        ],
        'Factor': [
            '0.21 kg/km', '0.17 kg/km', '0.12 kg/km', '0.05 kg/km',
            '0.089 kg/km', '0.255 kg/km', '0.195 kg/km',
            '0.42 kg/kWh', '2.0 kg/m³',
            '7.2 kg/day', '3.8 kg/day', '2.9 kg/day'
        ],
        'Source': [
            'EPA', 'EPA', 'EPA', 'EPA',
            'DEFRA', 'DEFRA', 'DEFRA',
            'IEA Global Average', 'IPCC',
            'Poore & Nemecek 2018', 'Poore & Nemecek 2018', 'Poore & Nemecek 2018'
        ]
    }
    
    st.dataframe(pd.DataFrame(factors_data), use_container_width=True)

def render_chatbot_tab():
    """Render the AI chatbot assistant tab."""

    st.subheader("🤖 Ask Your Carbon Assistant")
    st.markdown(
        '<p style="color: #94a3b8;">Ask anything about your footprint, what changes matter most, '
        'or why certain habits have such a big impact.</p>',
        unsafe_allow_html=True
    )

    # Initialise chat history in session state
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    user_data = st.session_state.get('user_data', {})
    results = st.session_state.get('results', {})

    # Show a banner if calculator not done yet
    if not st.session_state.get('calculated'):
        st.info("💡 Complete the calculator first so I can give you personalised answers — "
                "but you can still ask general carbon questions below!")

    # Suggested questions (only show if chat is empty)
    if not st.session_state.chat_messages:
        st.markdown("**Try asking:**")
        cols = st.columns(3)
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            with cols[i % 3]:
                if st.button(question, key=f"suggested_{i}", use_container_width=True):
                    st.session_state.chat_messages.append(
                        {"role": "user", "content": question}
                    )
                    with st.spinner("Thinking..."):
                        reply = get_chat_response(
                            st.session_state.chat_messages,
                            user_data,
                            results
                        )
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": reply}
                    )
                    st.rerun()

    st.markdown("---")

    # Render chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about your carbon footprint..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = get_chat_response(
                    st.session_state.chat_messages,
                    user_data,
                    results
                )
            st.markdown(reply)

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})

    # Clear chat button
    if st.session_state.chat_messages:
        st.markdown("---")
        if st.button("🗑️ Clear conversation", type="secondary"):
            st.session_state.chat_messages = []
            st.rerun()

def main():
    """Main application entry point."""

    # Train ML model on first run (cached to disk after that)
    if 'ml_ready' not in st.session_state:
        with st.spinner("🤖 Initialising ML model (first run only, ~30 seconds)..."):
            from ml_model import load_or_train_model
            load_or_train_model()
            st.session_state.ml_ready = True

    init_session_state()
    render_header()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Calculator",
        "📊 Results & Predictions",
        "💡 Reduction Tips",
        "🤖 AI Assistant",
        "ℹ️ About"
    ])
    
    with tab1:
        render_calculator_tab()
    
    with tab2:
        render_results_tab()
    
    with tab3:
        render_tips_tab()

    with tab4:
        render_chatbot_tab()
    
    with tab5:
        render_about_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #64748b; font-size: 0.8rem;">'
        'Built with ❤️ by Jyotismita | Data sources: EPA, DEFRA, IPCC'
        '</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

