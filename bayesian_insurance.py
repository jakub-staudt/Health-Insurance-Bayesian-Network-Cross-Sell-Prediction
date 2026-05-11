import streamlit as st
import graphviz
import pandas as pd

st.set_page_config(
    page_title="Bayesian Insurance Decision Network",
    layout="centered"
)

# -----------------------------
# Bayesian probability weights
# -----------------------------

vehicle_damage_probs = {
    "No": 0.005204,
    "Yes": 0.237655
}

previously_insured_probs = {
    "No": 0.225454,
    "Yes": 0.000905
}

vehicle_age_probs = {
    "< 1 Year": 0.043705,
    "1-2 Year": 0.173755,
    "> 2 Years": 0.293746
}

age_group_probs = {
    "18-24": 0.035294,
    "25-34": 0.088857,
    "35-44": 0.215932,
    "45-54": 0.201757,
    "55-64": 0.145770,
    "65+": 0.086196
}

prior_response_prob = 0.1226

vehicle_damage_given_age = {
    "1-2 Year": {
        "No": 0.359886,
        "Yes": 0.640114
    },
    "< 1 Year": {
        "No": 0.707524,
        "Yes": 0.292476
    },
    "> 2 Years": {
        "No": 0.000937,
        "Yes": 0.999063
    }
}


# -----------------------------
# Functions
# -----------------------------

def calculate_response_probability(
    vehicle_damage,
    previously_insured,
    vehicle_age,
    age_group
):
    response_probs = [
        vehicle_damage_probs[vehicle_damage],
        previously_insured_probs[previously_insured],
        age_group_probs[age_group]
    ]

    score = prior_response_prob

    for p in response_probs:
        score *= (p / prior_response_prob)

    # Dependency: Vehicle_Age affects Vehicle_Damage
    p_damage_given_age = vehicle_damage_given_age[vehicle_age][vehicle_damage]

    score *= p_damage_given_age

    score = max(0, min(score, 1))

    return score

def expected_utility(prob_response, profit_if_sale, marketing_cost):
    eu_offer = (prob_response * profit_if_sale) - marketing_cost
    eu_no_offer = 0

    return eu_offer, eu_no_offer


def create_bayesian_graph(
    vehicle_damage,
    previously_insured,
    vehicle_age,
    age_group,
    prob_response,
    eu_offer,
    eu_no_offer
):
    graph = graphviz.Digraph()

    graph.attr(rankdir="LR")
    graph.attr("node", shape="ellipse", style="filled", fillcolor="lightblue")

    graph.node("Age_Group", f"Age Group\n{age_group}")
    graph.node("Vehicle_Age", f"Vehicle Age\n{vehicle_age}")
    graph.node("Vehicle_Damage", f"Vehicle Damage\n{vehicle_damage}")
    graph.node("Previously_Insured", f"Previously Insured\n{previously_insured}")

    graph.node(
        "Response",
        f"Response\nP(yes) = {prob_response:.2%}",
        fillcolor="lightyellow"
    )

    graph.attr("node", shape="box", style="filled", fillcolor="lightgreen")
    graph.node("Decision", "Decision\nOffer Promotion?")

    graph.attr("node", shape="diamond", style="filled", fillcolor="lightpink")
    graph.node(
        "Utility",
        f"Utility\nEU(Offer) = {eu_offer:.2f}\nEU(No Offer) = {eu_no_offer:.2f}"
    )

    graph.edge("Age_Group", "Response")
    graph.edge("Vehicle_Age", "Vehicle_Damage")
    graph.edge("Vehicle_Damage", "Response")
    graph.edge("Previously_Insured", "Response")
    graph.edge("Response", "Utility")
    graph.edge("Decision", "Utility")

    return graph


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Bayesian Decision Network")
st.subheader("Health Insurance Cross-Sell Prediction")

st.write(
    "This tool estimates whether a customer should receive a vehicle insurance promotion "
    "based on Bayesian-style probability weights from the dataset."
)

st.divider()

st.header("Customer Information")

vehicle_damage = st.selectbox("Vehicle Damage", ["No", "Yes"])

previously_insured = st.selectbox("Previously Insured", ["No", "Yes"])

vehicle_age = st.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])

age_group = st.selectbox(
    "Age Group",
    ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
)

st.divider()

st.header("Business Utility Assumptions")

profit_if_sale = st.number_input(
    "Profit if customer buys insurance",
    min_value=0,
    value=500,
    step=50
)

marketing_cost = st.number_input(
    "Marketing cost of sending offer",
    min_value=0,
    value=50,
    step=10
)

st.divider()

# -----------------------------
# Calculations
# -----------------------------

prob_response = calculate_response_probability(
    vehicle_damage,
    previously_insured,
    vehicle_age,
    age_group
)

eu_offer, eu_no_offer = expected_utility(
    prob_response,
    profit_if_sale,
    marketing_cost
)

# -----------------------------
# Simple VPI for Previously_Insured
# -----------------------------

def simple_vpi_previously_insured(profit_if_sale, marketing_cost):
    # Prior probabilities of states
    p_not_insured = 0.54179
    p_insured = 0.45821

    # Conditional probabilities P(Response=1 | Previously_Insured)
    p_response_not_insured = 0.225454
    p_response_insured = 0.000905

    # Baseline: decision without knowing Previously_Insured
    p_response_baseline = 0.122563

    eu_offer_baseline = (p_response_baseline * profit_if_sale) - marketing_cost
    eu_no_offer_baseline = 0

    best_eu_without_info = max(eu_offer_baseline, eu_no_offer_baseline)

    # If we know customer is NOT previously insured
    eu_offer_not_insured = (p_response_not_insured * profit_if_sale) - marketing_cost
    best_eu_not_insured = max(eu_offer_not_insured, 0)

    # If we know customer IS previously insured
    eu_offer_insured = (p_response_insured * profit_if_sale) - marketing_cost
    best_eu_insured = max(eu_offer_insured, 0)

    # Expected utility when information is known
    best_eu_with_info = (
        p_not_insured * best_eu_not_insured
        + p_insured * best_eu_insured
    )

    vpi = best_eu_with_info - best_eu_without_info

    return vpi, best_eu_without_info, best_eu_with_info



# -----------------------------
# Outputs
# -----------------------------

st.header("Bayesian Output")

st.metric(
    label="Estimated P(Response = 1)",
    value=f"{prob_response:.2%}"
)

st.write("### Expected Utility")

st.write(f"EU(Offer Promotion) = **{eu_offer:.2f}**")
st.write(f"EU(No Promotion) = **{eu_no_offer:.2f}**")

st.divider()

st.header("Visual Bayesian Decision Network")

bayesian_graph = create_bayesian_graph(
    vehicle_damage,
    previously_insured,
    vehicle_age,
    age_group,
    prob_response,
    eu_offer,
    eu_no_offer
)

st.graphviz_chart(bayesian_graph)

st.divider()

st.header("Probability Evidence Table")

evidence_df = pd.DataFrame({
    "Variable": [
        "Vehicle Damage",
        "Previously Insured",
        "Vehicle Age",
        "Age Group"
    ],
    "Selected State": [
        vehicle_damage,
        previously_insured,
        vehicle_age,
        age_group
    ],
    "P(Response = 1 | State)": [
        vehicle_damage_probs[vehicle_damage],
        previously_insured_probs[previously_insured],
        vehicle_damage_given_age[vehicle_age][vehicle_damage],
        age_group_probs[age_group]
    ],
    "Meaning": [
        "P(Response = 1 | Vehicle_Damage)",
        "P(Response = 1 | Previously_Insured)",
        "P(Vehicle_Damage | Vehicle_Age)",
        "P(Response = 1 | Age_Group)"
    ]
})

st.dataframe(evidence_df, use_container_width=True)

st.divider()

st.header("Decision Recommendation")
st.write(f"Based on EU, a simple profit-loss fininacial comparison:")
if eu_offer > eu_no_offer:
    st.success("Recommended decision: Send promotion (eu_offer > eu_no_offer)")
else:
    st.error("Recommended decision: Do not send promotion (eu_offer < eu_no_offer)")

st.divider()

st.header("Explanation")

st.write(f"""
Selected evidence:

- Vehicle Damage: **{vehicle_damage}**
- Previously Insured: **{previously_insured}**
- Vehicle Age: **{vehicle_age}**
- Age Group: **{age_group}**

The model combines conditional probabilities from the dataset and estimates the probability
that the customer will respond positively to the insurance offer.

The decision is based on expected utility:

EU(Offer) = P(Response = 1) × Profit − Marketing Cost
""")

vpi_insured, eu_without_info, eu_with_info = simple_vpi_previously_insured(
    profit_if_sale,
    marketing_cost
)

st.divider()
st.header("Dataset-level VPI for Previously_Insured")

st.write("Variable tested: **Previously Insured**")

st.write(f"Best EU without information = **{eu_without_info:.2f}**")
st.write(f"Best EU with perfect information = **{eu_with_info:.2f}**")
st.write(f"VPI(Previously Insured) = **{vpi_insured:.2f}**")
st.write(f"*This VPI is computed for the whole customer population, not only for the selected customer profile.")
