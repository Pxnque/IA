import pandas as pd

def convert():

    file_path = 'dataset.csv'
    df = pd.read_csv(file_path)
    tema_deseado = "Generación Z y crisis de sentido"
    df_filtrado = df[df["tema"] == tema_deseado]
    textos = df_filtrado["texto"]
    with open("textoZ.txt", "w", encoding="utf-8") as f:
        for texto in textos:
            f.write(str(texto) + "\n")

if __name__ == "__main__":
    convert()