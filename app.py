import streamlit as st
import pandas as pd
import joblib
import sklearn
import category_encoders
import plotly.express as px

Model = joblib.load("Model.pkl")
Inputs = joblib.load("Inputs.pkl")
df = joblib.load("data.pkl")

def Make_Prdiction(online_order,book_table,location,approx_cost,listed_in,listed_in_city,rest_type,cuisines,Name_repeated):
    Pr_df = pd.DataFrame(columns=Inputs)
    Pr_df.at[0,"online_order"] = online_order
    Pr_df.at[0,"book_table"] = book_table
    Pr_df.at[0,"location"] = location
    Pr_df.at[0,"rest_type"] = rest_type
    Pr_df.at[0,"approx_cost(for two people)"] = approx_cost
    Pr_df.at[0,"cuisines"] = cuisines
    Pr_df.at[0,"listed_in(type)"] = listed_in
    Pr_df.at[0,"listed_in(city)"] = listed_in_city
    Pr_df.at[0,"Name_repeated"] = Name_repeated
    
    result = Model.predict(Pr_df)
    return result[0]
    
def main():
    st.title("Zomato Resturants")
    with st.sidebar:
        online_order= st.selectbox("online_order",['Yes', 'No']) 
        book_table = st.selectbox("book_table" ,['Yes', 'No'] )
        location = st.selectbox("location" ,['Banashankari', 'Basavanagudi', 'other', 'Jayanagar', 'JP Nagar',
           'Bannerghatta Road', 'BTM', 'Electronic City', 'Shanti Nagar',
           'Koramangala 5th Block', 'Richmond Road', 'HSR',
           'Koramangala 7th Block', 'Bellandur', 'Sarjapur Road',
           'Marathahalli', 'Whitefield', 'Old Airport Road', 'Indiranagar',
           'Koramangala 1st Block', 'Frazer Town', 'MG Road', 'Brigade Road',
           'Lavelle Road', 'Church Street', 'Ulsoor', 'Residency Road',
           'Shivajinagar', 'St. Marks Road', 'Cunningham Road',
           'Commercial Street', 'Vasanth Nagar', 'Domlur',
           'Koramangala 8th Block', 'Ejipura', 'Jeevan Bhima Nagar',
           'Kammanahalli', 'Koramangala 6th Block', 'Brookefield',
           'Koramangala 4th Block', 'Banaswadi', 'Kalyan Nagar',
           'Malleshwaram', 'Rajajinagar', 'New BEL Road'] )
        rest_type = st.selectbox("rest_type " ,['Casual Dining', 'Cafe, Casual Dining', 'Quick Bites',
           'Casual Dining, Cafe', 'Cafe', 'Quick Bites, Cafe', 'Delivery',
           'Dessert Parlor', 'Pub', 'Beverage Shop', 'Bar',
           'Takeaway, Delivery', 'Food Truck', 'Quick Bites, Dessert Parlor',
           'Pub, Casual Dining', 'Casual Dining, Bar', 'Bakery', 'Sweet Shop',
           'Dessert Parlor, Beverage Shop', 'Beverage Shop, Quick Bites',
           'Microbrewery, Casual Dining', 'Sweet Shop, Quick Bites', 'Lounge',
           'Food Court', 'Cafe, Bakery', 'Microbrewery', 'Kiosk', 'Pub, Bar',
           'Casual Dining, Pub', 'Cafe, Quick Bites', 'Lounge, Bar',
           'Bakery, Quick Bites', 'Dessert Parlor, Quick Bites',
           'Bar, Casual Dining', 'Beverage Shop, Dessert Parlor',
           'Casual Dining, Microbrewery', 'Mess', 'Lounge, Casual Dining',
           'Cafe, Dessert Parlor', 'Dessert Parlor, Cafe',
           'Bakery, Dessert Parlor', 'Quick Bites, Sweet Shop', 'Takeaway',
           'Microbrewery, Pub', 'Club', 'Fine Dining', 'Bakery, Cafe',
           'Beverage Shop, Cafe', 'Pub, Cafe', 'Casual Dining, Irani Cafee',
           'Food Court, Quick Bites', 'Quick Bites, Beverage Shop',
           'Fine Dining, Lounge', 'Quick Bites, Bakery', 'Bar, Quick Bites',
           'Pub, Microbrewery', 'Microbrewery, Lounge',
           'Fine Dining, Microbrewery', 'Fine Dining, Bar',
           'Dessert Parlor, Kiosk', 'Cafe, Bar', 'Quick Bites, Food Court',
           'Casual Dining, Lounge', 'Microbrewery, Bar', 'Cafe, Lounge',
           'Bar, Pub', 'Lounge, Cafe', 'Dessert Parlor, Bakery',
           'Club, Casual Dining', 'Lounge, Microbrewery', 'Dhaba',
           'Bar, Lounge', 'Food Court, Casual Dining'] )
        approx_cost = st.number_input("approx_cost:", min_value=10, max_value=10000, step=1, value=10)
        cuisines = st.selectbox("cuisine types" ,['North Indian, Mughlai, Chinese', 'Chinese, North Indian, Thai',
           'Cafe, Mexican, Italian', ..., 'Andhra, Hyderabadi, Biryani',
           'Andhra, North Indian, South Indian', 'Thai, Chinese, Momos'] )
        listed_in =  st.selectbox("listed_in_types" ,['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out','Drinks & nightlife', 'Pubs and bars'] )
        listed_in_city = st.selectbox("listed_in_city" ,['Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
           'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
           'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
           'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
           'Koramangala 4th Block', 'Koramangala 5th Block',
           'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
           'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',
           'Old Airport Road', 'Rajajinagar', 'Residency Road',
           'Sarjapur Road', 'Whitefield'] )
        Name_repeated = st.slider("How many Resturants do yo have" ,  min_value=1, max_value=100, value=1, step=1)
    if st.button("Predict"):
        px.histogram(df,y = "votes", x = 'listed_in(city)')
        Results = Make_Prdiction(online_order,book_table,location,approx_cost,listed_in,listed_in_city,rest_type,cuisines,Name_repeated)
        list_success = ["Your Resturant May Fail" , "Your Resturant will success"]
        st.text(list_success[Results])
main()

