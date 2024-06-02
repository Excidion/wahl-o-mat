# european parliament 2024
wget https://www.bpb.de/system/files/datei/Wahl-O-Mat_Europa_2024_Datensatz.zip -P ./data/raw
unzip ./data/raw/Wahl-O-Mat_Europa_2024_Datensatz.zip -d ./data/interim/2024
# bundestag 2021
wget https://www.bpb.de/system/files/datei/Wahl-O-Mat%20Bundestag%202021_Datensatz_v1.02.zip -P ./data/raw
unzip "data/raw/Wahl-O-Mat Bundestag 2021_Datensatz_v1.02.zip" -d ./data/interim/2021
