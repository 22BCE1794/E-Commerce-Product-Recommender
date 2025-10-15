import streamlit as st
import requests
st.set_page_config(page_title='E-commerce Recommender Demo', layout='centered')

st.title('E-commerce Recommender - Demo (OpenAI mode)')

api_url = st.text_input('Backend URL', value='http://localhost:8000')

user_id = st.text_input('User ID', value='u1')
limit = st.slider('Top N', 1, 10, 5)
tone = st.selectbox('Tone', ['friendly','formal','promotional'])

if st.button('Get Recommendations'):
    payload = {'user_id': user_id, 'limit': limit, 'tone': tone}
    try:
        r = requests.post(f"{api_url}/recommend", json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
        for rec in data.get('recommendations', []):
            prod = rec['product']
            st.subheader(prod.get('title'))
            st.write(f"**Explanation:** {rec.get('explanation')}" )
            st.write(f"Category: {prod.get('category')} • Brand: {prod.get('brand')} • Price: ${prod.get('price')}")
            st.divider()
    except Exception as e:
        st.error(f"API error: {e}")
