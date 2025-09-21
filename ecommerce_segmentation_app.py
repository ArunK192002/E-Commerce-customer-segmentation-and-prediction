import streamlit as st
import pandas as pd
import joblib
import datetime

st.set_page_config(page_title="Ecommerce Customer Segmentation and Prediction")
st.title("🛍️ Ecommerce Customer Segmentation and Prediction")

@st.cache_resource
def load_model():
    model = joblib.load("tuned_gradient_boosting_model.pkl")
    le_country = joblib.load("label_encoder_country.pkl")
    le_segment = joblib.load("label_encoder_segment.pkl")
    return model, le_country, le_segment

model, le_country, le_segment = load_model()

if 'entries' not in st.session_state:
    st.session_state['entries'] = []

def engineer_features(ecom_df):
    ecom_df['TotalPrice'] = ecom_df['UnitPrice'] * ecom_df['Quantity']
    ecom_df['IsCancelled'] = ecom_df['InvoiceNo'].astype(str).apply(lambda x: 1 if x.startswith('C') else 0)
    grouped = ecom_df.groupby('CustomerID')
    reference_date = ecom_df['InvoiceDate'].max()

    recency = (reference_date - grouped['InvoiceDate'].max()).dt.days
    frequency = grouped['InvoiceNo'].nunique()
    monetary = grouped['TotalPrice'].sum()
    avg_basket_size = grouped['Quantity'].sum() / frequency
    avg_order_value = monetary / frequency
    product_count = grouped['StockCode'].nunique()
    total_qty = grouped['Quantity'].sum()
    first_date = grouped['InvoiceDate'].min()
    last_date = grouped['InvoiceDate'].max()
    lifetime = (last_date - first_date).dt.days
    avg_days_btwn_orders = grouped['InvoiceDate'].apply(lambda x: x.sort_values().diff().dt.days.mean())
    avg_unit_price = grouped['UnitPrice'].mean()
    cancelled_orders = ecom_df[ecom_df['IsCancelled'] == 1].groupby('CustomerID')['InvoiceNo'].nunique()
    return_rate = (cancelled_orders / frequency).fillna(0)
    country = grouped['Country'].first()
    variety = ecom_df.groupby(['CustomerID', 'InvoiceNo'])['StockCode'].nunique().groupby('CustomerID').mean()

    def reorder_rate(series):
        counts = series.value_counts()
        return counts[counts > 1].sum() / counts.sum() if counts.sum() > 0 else 0

    reorder = ecom_df.groupby('CustomerID')['StockCode'].apply(reorder_rate)

    df = pd.DataFrame({
        'CustomerID': recency.index,
        'RecencyInDays': recency.values,
        'TotalOrderFrequency': frequency.values,
        'TotalMonetaryValue': monetary.values,
        'AverageBasketSize': avg_basket_size.values,
        'AverageOrderValue': avg_order_value.values,
        'DistinctProductCount': product_count.values,
        'TotalQuantityPurchased': total_qty.values,
        'FirstPurchaseDate': first_date.values,
        'LastPurchaseDate': last_date.values,
        'CustomerLifetimeInDays': lifetime.values,
        'AverageDaysBetweenOrders': avg_days_btwn_orders.fillna(0).values,
        'AverageUnitPrice': avg_unit_price.values,
        'ReturnRate': return_rate.values,
        'CustomerCountry': country.values,
        'AverageProductVarietyPerOrder': variety.values,
        'FrequentReorderRate': reorder.values
    })

    return df

st.subheader("📅 Enter Single Transaction")

with st.form("entry_form"):
    invoice = st.text_input("Invoice No")
    stock = st.text_input("Stock Code")
    desc = st.text_input("Description")
    qty = st.number_input("Quantity", min_value=1)
    price = st.number_input("Unit Price", min_value=0.01, format="%.2f")
    cust_id = st.text_input("Customer ID")

    date_part = st.date_input("Invoice Date")
    time_part = st.time_input("Invoice Time", value=datetime.time(hour=0, minute=0))
    invoice_datetime = datetime.datetime.combine(date_part, time_part)

    country = st.selectbox("Country", sorted([
        'Australia', 'Austria', 'Bahrain', 'Belgium', 'Brazil', 'Canada', 'Channel Islands',
        'Cyprus', 'Czech Republic', 'Denmark', 'EIRE', 'European Community', 'Finland',
        'France', 'Germany', 'Greece', 'Hong Kong', 'Iceland', 'Israel', 'Italy', 'Japan',
        'Lebanon', 'Lithuania', 'Malta', 'Netherlands', 'Norway', 'Poland', 'Portugal',
        'RSA', 'Saudi Arabia', 'Singapore', 'Spain', 'Sweden', 'Switzerland',
        'United Arab Emirates', 'United Kingdom', 'Unspecified', 'USA'
    ]))
    next_entry = st.form_submit_button("Add Entry")

if next_entry:
    try:
        if not cust_id.strip():
            raise ValueError("Customer ID is required.")
        if price <= 0:
            raise ValueError("Unit Price must be greater than 0.")
        if qty <= 0:
            raise ValueError("Quantity must be greater than 0.")

        invoice = invoice.strip() if invoice.strip() else "UNKNOWN"
        stock = stock.strip() if stock.strip() else "00000"
        desc = desc.strip() if desc.strip() else "No Description"

        st.session_state['entries'].append({
            "InvoiceNo": invoice,
            "StockCode": stock,
            "Description": desc,
            "Quantity": int(qty),
            "InvoiceDate": invoice_datetime,
            "UnitPrice": float(price),
            "CustomerID": float(cust_id),
            "Country": country
        })

        st.success("✅ Entry added.")

    except Exception as e:
        st.error(f"❌ Invalid input: {e}")

if st.session_state['entries']:
    st.subheader("📋 Preview Entries")
    st.dataframe(pd.DataFrame(st.session_state['entries']))

    if st.button("✅ Submit for Prediction"):
        try:
            ecom_df = pd.DataFrame(st.session_state['entries'])
            ecom_df['InvoiceDate'] = pd.to_datetime(ecom_df['InvoiceDate'], errors='coerce')
            seg_df = engineer_features(ecom_df)
            seg_df['CustomerCountry'] = le_country.transform(seg_df['CustomerCountry'])
            features = seg_df.drop(columns=['CustomerID', 'FirstPurchaseDate', 'LastPurchaseDate'])
            preds = model.predict(features)
            labels = le_segment.inverse_transform(preds)
            seg_df['SegmentCode'] = preds
            seg_df['SegmentPrediction'] = labels
            st.success("✅ Prediction Completed")
            st.dataframe(seg_df)

            csv = seg_df.to_csv(index=False).encode('utf-8')
            st.download_button("📅 Download Results", data=csv, file_name="predicted_segments.csv", mime='text/csv')
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")

st.subheader("📁 Or Upload Transactions CSV/XLSX")
uploaded = st.file_uploader("Upload File", type=["csv", "xlsx"])

if uploaded:
    try:
        if uploaded.name.endswith('.csv'):
            ecom_df = pd.read_csv(uploaded, encoding='ISO-8859-1')
        else:
            ecom_df = pd.read_excel(uploaded, encoding='ISO-8859-1')

        ecom_df['InvoiceDate'] = pd.to_datetime(ecom_df['InvoiceDate'], errors='coerce')

        ecom_df.dropna(subset=['InvoiceDate', 'UnitPrice', 'Quantity', 'CustomerID'], inplace=True)
        ecom_df['InvoiceNo'].fillna('UNKNOWN', inplace=True)
        ecom_df['StockCode'].fillna('00000', inplace=True)
        ecom_df['Description'].fillna('No Description', inplace=True)
        ecom_df['Country'].fillna('Unknown', inplace=True)

        seg_df = engineer_features(ecom_df)
        seg_df['CustomerCountry'] = le_country.transform(seg_df['CustomerCountry'])
        features = seg_df.drop(columns=['CustomerID', 'FirstPurchaseDate', 'LastPurchaseDate'])
        preds = model.predict(features)
        labels = le_segment.inverse_transform(preds)
        seg_df['SegmentCode'] = preds
        seg_df['SegmentPrediction'] = labels

        st.success("✅ Prediction Completed for Uploaded File")
        st.dataframe(seg_df)

        csv = seg_df.to_csv(index=False).encode('utf-8')
        st.download_button("📅 Download Results", data=csv, file_name="predicted_segments.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ Error: {e}")
