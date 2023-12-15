#!/bin/sh
python backend_flask.py & streamlit run frontend_streamlit.py

# # Lancement du backend Flask avec Gunicorn
# gunicorn -w 4 -b 0.0.0.0:5000 backend_flask:app &

# # Attendez que le backend soit prêt (ajuster le délai si nécessaire)
# sleep 5

# # Lancement du frontend Streamlit
# streamlit run frontend_streamlit.py