import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

# Load model and scaler
model = joblib.load("rfm_classifier.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Segmentation Predictor")

# Mode selection
analysis_mode = st.radio("Select analysis mode:", 
                        ("Batch CSV Processing", "Single Record Prediction"))

if analysis_mode == "Batch CSV Processing":
    # Batch processing code (your existing implementation)
    uploaded_file = st.file_uploader("Upload customer data (.CSV)", type=["csv"])

    if uploaded_file is not None:
        # Read and process data
        df = pd.read_csv(uploaded_file, parse_dates=['InvoiceDate'])
        
        # Data validation
        required_cols = ['InvoiceNo', 'CustomerID', 'InvoiceDate', 'TotalPrice']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
        else:
            with st.spinner('Analyzing transactions...'):
                # Calculate RFM features
                snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
                rfm = df.groupby('CustomerID').agg({
                    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
                    'InvoiceNo': 'nunique',
                    'TotalPrice': ['sum', 'mean']
                }).reset_index()
                
                # Flatten multi-index columns
                rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'AOV']
                
                # Scale features and predict
                features = scaler.transform(rfm[['Recency', 'Frequency', 'Monetary', 'AOV']])
                segments = model.predict(features)
                
                # Add segments to both transaction and RFM data
                segment_map = {
                    0: "High-Value Low-Frequency",
                    1: "Inactive Low-Spenders",
                    2: "Stable Medium-Spenders",
                    3: "Loyal High-Spenders"
                }
                
                rfm['Segment'] = segments
                rfm['Segment_Label'] = rfm['Segment'].map(segment_map)
                
                # Merge back with original transactions
                result_df = pd.merge(df, rfm[['CustomerID', 'Segment', 'Segment_Label']], 
                                    on='CustomerID', how='left')
                
                # Show results
                st.success(f"Analyzed {len(df)} transactions for {len(rfm)} customers")
                
                # Single tab for Transaction Data
                tab1 = st.tabs(["Transaction Data"])[0]  # Get first tab
                
                with tab1:
                    st.write("Enhanced transaction data with segments:")
                    st.dataframe(result_df.head())
                
                # Single download button
                st.download_button(
                    label="Download enhanced transactions",
                    data=result_df.to_csv(index=False).encode('utf-8'),
                    file_name='segmented_transactions.csv',
                    mime='text/csv'
                )
else:  # Single Record Prediction
    st.subheader("Enter Customer RFM Values")
    
    # Display cluster statistics for reference
    with st.expander("📊 Cluster Statistics Reference"):
        st.write("""
        | Metric       | Cluster 0          | Cluster 1          | Cluster 2          | Cluster 3          |
        |--------------|--------------------|--------------------|--------------------|--------------------|
        | Frequency    | 1.0 (One-time)     | 1.0-5.0            | 5.0+               | 10.0+              |
        | Monetary     | $1757+ (High AOV)  | $167-$649          | $649-$1619         | $1619+             |
        | Recency      | <30 days           | 30-142 days        | 18-142 days        | <18 days           |
        | AOV          | $1757+             | $172-$281          | $281-$420          | $420+              |
        """)
        st.caption("Based on dataset (Online Retail) statistics (25th-75th percentiles)")

    # Create input fields
    col1, col2 = st.columns(2)
    with col1:
        recency = st.number_input("Recency (days)", min_value=0, 
                                 help="Days since last purchase (Avg: 92.5)")
        frequency = st.number_input("Frequency (# orders)", min_value=1, 
                                  help="Typical range: 1-210 (Avg: 4.3)")
    with col2:
        monetary = st.number_input("Monetary ($ total)", min_value=0.0, 
                                 help="Total spending (Avg: $1910)")
        aov = st.number_input("AOV ($ average)", min_value=0.0, 
                            help="Average order value (Avg: $364)")
    
    if st.button("Predict Segment"):
        try:
            # Prepare and scale input
            input_data = scaler.transform([[recency, frequency, monetary, aov]])
            
            # Predict
            segment = model.predict(input_data)[0]
            
            #segment insights dictionary
            segment_insights = {
                0: {
                    "label": "High-Value Low-Frequency",
                    "desc": "Premium one-time buyers",
                    "stats": "1 purchase, $1757+ AOV, <30 days recency",
                    "action": "Upsell complementary products, request feedback"
                },
                1: {
                    "label": "Inactive Low-Spenders",
                    "desc": "Dormant budget shoppers",
                    "stats": "Typically: 1-5 orders, $167-$649 spend, 30-142 days recency",
                    "action": "Win-back campaigns, discount offers"
                },
                2: {
                    "label": "Stable Medium-Spenders",
                    "desc": "Regular loyal customers",
                    "stats": "5+ orders, $649-$1619 spend, 18-142 days recency",
                    "action": "Loyalty rewards, subscription offers"
                },
                3: {
                    "label": "Loyal High-Spenders",
                    "desc": "Frequent big spenders",
                    "stats": "10+ orders, $1619+ spend, <18 days recency",
                    "action": "VIP treatment"
                }
            }
            
            insight = segment_insights[segment]
            
            # Display results
            st.success("## Prediction Results")
            
            #printing insights in columns
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Segment**")
                st.info(f"{segment} - {insight['label']}")
                
                st.markdown("**Description**")
                st.info(insight["desc"])
            with col2:
                st.caption("**Typical Profile**")
                st.info(insight["stats"])
                st.caption("**Recommended Action**")
                st.info(insight["action"])
            
            # Comparison visualization
            st.subheader("How You Compare")
            
            # Create comparison data
            comparison_data = pd.DataFrame({
                'Metric': ['Recency', 'Frequency', 'Monetary', 'AOV'],
                'Your Value': [recency, frequency, monetary, aov],
                'Segment Avg': [
                    [30, 92.5, 142][min(segment, 2)],  # Recency
                    [1, 4.3, 10][min(segment, 2)],     # Frequency
                    [1757, 1910, 1619][min(segment, 2)], # Monetary
                    [1757, 364, 420][min(segment, 2)] # AOV
                ]
            })
            
            st.bar_chart(
                comparison_data.set_index('Metric'),
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")