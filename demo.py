import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
from skimage import io
import numpy as np
import matplotlib.pyplot as plt



st.title("Sars-Cov recognizer ")
st.header('This application helps to identify the existence of SARS virus based on X-ray images of the lungs')
#inf est la variable choisis
image = Image.open(r"C:\Users\MonPc\Desktop\prjct\1200px-Novel_Coronavirus_SARS-CoV-2.jpg")
st.sidebar.image(image, use_column_width=True)
st.sidebar.title('MORE INFORMATIONS')
inf = st.sidebar.selectbox(
    "What would you like to know about Sars-Cov virus?",
    ("(Please select a category)", "Overview", "Symptoms","Treatment")
)
if inf == "Overview" :
    rep = st.sidebar.title ( "Overview" )
    rep = st.sidebar.subheader("According to World Health Organization")
    rep = st.sidebar.write('Severe acute respiratory syndrome (SARS) is a viral respiratory disease caused by a SARS-associated coronavirus. It was first identified at the end of February 2003 during an outbreak that emerged in China and spread to 4 other countries. WHO co-ordinated the international investigation with the assistance of the Global Outbreak Alert and Response Network (GOARN) and worked closely with health authorities in affected countries to provide epidemiological, clinical and logistical support and to bring the outbreak under control.')
    rep = st.sidebar.write('SARS is an airborne virus and can spread through small droplets of saliva in a similar way to the cold and influenza. It was the first severe and readily transmissible new disease to emerge in the 21st century and showed a clear capacity to spread along the routes of international air travel.')
    rep = st.sidebar.write('SARS can also be spread indirectly via surfaces that have been touched by someone who is infected with the virus.')
    rep = st.sidebar.write('Most patients identified with SARS were previously healthy adults aged 25–70 years. A few suspected cases of SARS have been reported among children under 15 years. The case fatality among persons with illness meeting the current WHO case definition for probable and suspected cases of SARS is around 3%.')

elif inf == "Symptoms" :
    rep = st.sidebar.title( "Symptoms" )
    rep = st.sidebar.subheader( "According to World Health Organization" )
    rep = st.sidebar.write("The incubation period of SARS is usually 2-7 days but may be as long as 10 days.")
    rep = st.sidebar.write("The first symptom of the illness is generally fever (>38°C), which is often high, and sometimes associated with chills and rigors. It may also be accompanied by other symptoms including headache, malaise, and muscle pain. At the onset of illness, some cases have mild respiratory symptoms. Typically, rash and neurologic or gastrointestinal findings are absent, although a few patients have reported diarrhoea during the early febrile stage.")
    rep = st.sidebar.write("After 3-7 days, a lower respiratory phase begins with the onset of a dry, non-productive cough or dyspnoea (shortness of breath) that may be accompanied by, or progress to, hypoxemia (low blood oxygen levels). In 10–20% of cases, the respiratory illness is severe enough to require intubation and mechanical ventilation. Chest radiographs may be normal throughout the course of illness, though not for all patients. The white blood cell count is often decreased early in the disease, and many people have low platelet counts at the peak of the disease. ")

elif inf == "Treatment" :
    rep = st.sidebar.title( "Treatment" )
    rep = st.sidebar.subheader( "According to World Health Organization" )
    rep = st.sidebar.write('There is no cure or vaccine for SARS and treatment should be supportive and based on the patient’s symptoms.')
    rep = st.sidebar.write('Controlling outbreaks relies on containment measures including:')
    rep = st.sidebar.write('- Prompt detection of cases through good surveillance networks and including an early warning system;')
    rep = st.sidebar.write('- Isolation of suspected of probably cases;')
    rep = st.sidebar.write('- Tracing to identify both the source of the infection and contacts of those who are sick and may be at risk of contracting the virus;')
    rep = st.sidebar.write('- Quarantine of suspected contacts for 10 days;')
    rep = st.sidebar.write('- Exit screening for outgoing passengers from areas with recent local transmission by asking questions and temperature measurement;')
    rep = st.sidebar.write('- Disinfection of aircraft and cruise vessels having SARS cases on board using WHO guidelines.')
    rep = st.sidebar.write('Personal preventive measures to prevent spread of the virus include frequent hand washing using soap or alcohol-based disinfectants. For those with a high risk of contracting the disease, such as health care workers, use of personal protective equipment, including a mask, goggles and an apron is mandatory.  Whenever possible, household contacts should also wear a mask.')

st.write("""
_**Made by:**_ Chifaa & Imane  
""")

file = st.file_uploader( "Please upload your X-ray image for the lungs", type=["jpg","png","jpeg"])
if st.button('Show the results') and file is not None: 
    # test file with the trained model (à faire)
    model = load_model(r"C:\Users\MonPc\Desktop\prjct\keras_model.h5")
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    #image = Image.open('<IMAGE_PATH>') image = file
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = Image.open(file).convert('RGB')
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normlimgarray = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normlimgarray
    #plt.axis("off")
    # run the inference
    prediction = model.predict(data)
    #st.header(prediction)
    #st.header(type(prediction))
    #st.header(prediction.shape)
    ans = prediction[0][0]
    if ans >= 0.5:
        st.header('YOU ARE NOT AFFECTED')
        st.image(file)
    elif ans < 0.5:
        st.header('YOU ARE AFFECTED')
        st.image(file)
    st.write(model)