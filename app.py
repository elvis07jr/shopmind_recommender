import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ===============================
# Load artifacts
# ===============================
rules = pd.read_pickle('recommendation_rules.pkl')
te = joblib.load('transaction_encoder.pkl')

# ===============================
# App header
# ===============================
st.title("🧠 ShopMind: Smart Recommendation Engine")
st.write("Get recommendations based on how users like you behave on the site.")

# Small explainer for lift (user-friendly, no logic change)
with st.expander("ℹ️ What does 'Lift' mean?"):
    st.write(
        "Lift compares how likely a user is to purchase given a behavior pattern versus average. "
        "For example, a lift of **2.0x** means users with that behavior are **twice as likely** to purchase."
    )

# ===============================
# Input form
# ===============================
st.subheader("📝 Describe the User Session")

col1, col2 = st.columns(2)

with col1:
    admin_pages = st.number_input(
        "⚙️ Admin/Help Pages Visited",
        min_value=0, step=1,
        help="Pages like FAQs, account settings, or support."
    )
    info_pages = st.number_input(
        "📖 Info Pages Viewed",
        min_value=0, step=1,
        help="General info pages such as 'About Us' or 'Policies'."
    )
    product_pages = st.number_input(
        "🛒 Product Pages Visited",
        min_value=0, step=1,
        help="Pages showing products or services."
    )

with col2:
    admin_duration = st.number_input(
        "⏱️ Time on Admin/Help Pages (seconds)",
        min_value=0.0,
        help="How long the user spent on support/help sections."
    )
    prod_duration = st.number_input(
        "⏱️ Time on Product Pages (seconds)",
        min_value=0.0,
        help="How long the user spent viewing products."
    )

# ===============================
# Helper: binning (must match training)
# ===============================
def get_bin(value, bins, labels):
    for i, threshold in enumerate(bins):
        if value <= threshold:
            return labels[i]
    return labels[-1]

# User-friendly category labels for UI
admin_cat = get_bin(
    admin_pages, [0, 2, 5],
    ['No Admin Use', 'Low Admin Use', 'Medium Admin Use']
) if admin_pages <= 5 else 'High Admin Use'

info_cat = get_bin(
    info_pages, [0, 2, 5],
    ['No Info Use', 'Low Info Use', 'Medium Info Use']
) if info_pages <= 5 else 'High Info Use'

product_cat = get_bin(
    product_pages, [0, 2, 5],
    ['No Product Views', 'Low Product Views', 'Medium Product Views']
) if product_pages <= 5 else 'High Product Views'

admin_dur_cat = get_bin(
    admin_duration, [0, 50, 200],
    ['No Time on Admin Pages', 'Short Time on Admin Pages', 'Medium Time on Admin Pages']
) if admin_duration <= 200 else 'Long Time on Admin Pages'

prod_dur_cat = get_bin(
    prod_duration, [0, 100, 500],
    ['No Time on Product Pages', 'Short Time on Product Pages', 'Medium Time on Product Pages']
) if prod_duration <= 500 else 'Long Time on Product Pages'

# ===============================
# Mapping: UI-friendly → encoder labels (backend compatibility)
# ===============================
label_map = {
    'No Admin Use': 'Admin_None',
    'Low Admin Use': 'Admin_Low',
    'Medium Admin Use': 'Admin_Med',
    'High Admin Use': 'Admin_High',

    'No Info Use': 'Info_None',
    'Low Info Use': 'Info_Low',
    'Medium Info Use': 'Info_Med',
    'High Info Use': 'Info_High',

    'No Product Views': 'Product_None',
    'Low Product Views': 'Product_Low',
    'Medium Product Views': 'Product_Med',
    'High Product Views': 'Product_High',

    'No Time on Admin Pages': 'AdminDur_Zero',
    'Short Time on Admin Pages': 'AdminDur_Short',
    'Medium Time on Admin Pages': 'AdminDur_Med',
    'Long Time on Admin Pages': 'AdminDur_Long',

    'No Time on Product Pages': 'ProdDur_Zero',
    'Short Time on Product Pages': 'ProdDur_Short',
    'Medium Time on Product Pages': 'ProdDur_Med',
    'Long Time on Product Pages': 'ProdDur_Long'
}

# Reverse mapping for display of rule antecedents (encoder → friendly)
reverse_label_map = {v: k for k, v in label_map.items()}

# Assemble transaction
current_basket_friendly = [admin_cat, info_cat, product_cat, admin_dur_cat, prod_dur_cat]
current_basket = [label_map[x] for x in current_basket_friendly]  # safe for encoder

# ===============================
# Transform using encoder (unchanged logic)
# ===============================
encoded_input = te.transform([current_basket])
input_df = pd.DataFrame(encoded_input, columns=te.columns_)

# ===============================
# Find matching rules (unchanged logic)
# ===============================
matches = []
for _, rule in rules.iterrows():
    antecedents = set(rule['antecedents'])
    if antecedents.issubset(set(current_basket)):
        matches.append(rule)

# ===============================
# Session summary chips (visual only)
# ===============================
st.subheader("🧭 Session Summary")
summary_cols = st.columns(5)
chips = [
    admin_cat,
    info_cat,
    product_cat,
    admin_dur_cat,
    prod_dur_cat
]
for i, chip in enumerate(chips):
    with summary_cols[i]:
        st.markdown(
            f"<div style='padding:8px 10px; border-radius:999px; "
            f"border:1px solid #e5e7eb; display:inline-block; font-size:12px;'>"
            f"{chip}</div>",
            unsafe_allow_html=True
        )

# ===============================
# Display results (more interactive/visual)
# ===============================
st.subheader("🔮 Recommendations")

if len(matches) > 0:
    st.success("🎯 We found behavior patterns similar to this session!")

    # Show up to top 3 matches in the order they were found (logic unchanged)
    for i, match in enumerate(matches[:3]):
        antecedents_enc = list(match['antecedents'])
        # Map encoder labels to friendly labels for display
        antecedents_friendly = [reverse_label_map.get(a, a) for a in antecedents_enc]
        lift_value = float(match['lift'])

        with st.expander(f"📊 Pattern {i+1}: {', '.join(antecedents_friendly)}"):
            # Big metric for lift
            st.metric(label="✨ Likelihood to Purchase (Lift)", value=f"{lift_value:.2f}x")

            # Progress bar as a quick visual cue (capped at 10x for readability)
            bar_value = min(lift_value / 10.0, 1.0)
            st.progress(bar_value)

            # Suggested action (visual box)
            st.markdown(
                "<div style='padding:12px; background:#f0f9ff; border-radius:10px; border:1px solid #bae6fd;'>"
                "<b>👉 Suggested Action:</b> Offer a <b>discount</b>, <b>promo code</b>, or <b>live support chat</b>."
                "</div>",
                unsafe_allow_html=True
            )
else:
    st.warning("⚠️ No strong behavioral match found. Try general engagement tactics like free shipping banners or product highlights.")

# ===============================
# Sidebar: Top global rule (same data, nicer display)
# ===============================
st.sidebar.header("💡 Overall Best Insight")
top_rule = rules.iloc[0] if len(rules) > 0 else None
if top_rule is not None:
    ant_top_enc = list(top_rule['antecedents'])
    ant_top_friendly = [reverse_label_map.get(a, a) for a in ant_top_enc]
    st.sidebar.write("Most effective pattern observed:")
    st.sidebar.write(f"• {', '.join(ant_top_friendly)}")
    st.sidebar.metric("Likelihood to Purchase (Lift)", f"{float(top_rule['lift']):.2f}x")

    # Sidebar visual bar
    st.sidebar.progress(min(float(top_rule['lift']) / 10.0, 1.0))
