"""Realizar una “canalización” o “pipeline” para analizar el siguiente corpus CorpusLenguajes.txt
-Aplicar stop_word
-Lematización
-Tf-Idf
-Mostrar el corpus preparado
-Mostrar la matriz TF-IDF generada
-Mostrar el vocabulario generado
Analizar el mismo y redactar un informe con las conclusiones obtenidas.
-Obtener las jerarquía de 6 palabras mas usadas en todo el corpus
-La palabra menos utilizada
-Las palabras mas repetidas en la misma oración
-Imprimir el gráfico de Distribución de Frecuencia.
"""
#CANALIZACION COPIANDO LAS ORACIONES EN UNA VARIABLE Y TRABAJANDO SIN IMPORTAR DESDE ALGUN OTRO ARCHIVO
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
#1)Realizamos las importaciones de librerias que vamos a usar en todo el ejercicio
import nltk #Biblioteca principal que usamos
import string  # Para manejar puntuación
from nltk.tokenize import word_tokenize, sent_tokenize  # Tokenización de palabras y oraciones
from nltk.corpus import stopwords  # Stopwords para eliminar palabras comunes
from nltk.stem import WordNetLemmatizer  # Lematización para reducir palabras a su forma base
from nltk.corpus import wordnet  # Para mejorar la lematización
from nltk import FreqDist
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from sklearn.feature_extraction.text import TfidfVectorizer
# Descargar el modelo POS Tagger necesario
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')  # ESTE ES EL IMPORTANTE
nltk.download('wordnet')
nltk.download('omw-1.4')  # Opcional, mejora la lematización

oraciones=["Python is an interpreted and high-level language, while CPlus is a compiled and low-level language .-","JavaScript runs in web browsers, while Python is used in various applications, including data science and artificial intelligence.","JavaScript is dynamically and weakly typed, while Rust is statically typed and ensures greater data security .-","Python and JavaScript are interpreted languages, while Java, CPlus, and Rust require compilation before execution.","JavaScript is widely used in web development, while Go is ideal for servers and cloud applications.","Python is slower than CPlus and Rust due to its interpreted nature.","JavaScript has a strong ecosystem with Node.js for backend development, while Python is widely used in data science .-","JavaScript does not require compilation, while CPlus and Rust require code compilation before execution .-","Python and JavaScript have large communities and an extensive number of available libraries.","Python is ideal for beginners, while Rust and CPlus are more suitable for experienced programmers."]
#Unimos las oraciones en una sola cadena
texto = " ".join(oraciones)

#2)Definimos la función para tokenizar el texto, tokenizar el texto es separar palabra por palabra el texto en una lista
def tokenizar(texto):
    # Tokenizamos el texto en palabras
    tokens = word_tokenize(texto)
    return tokens
texto_tokenizado =tokenizar(texto)

#3)Definimos la función para quitar stopwords(quitamos palabras vacias)
def quitarStopwords_eng(texto):
    ingles = stopwords.words("english")
    texto_limpio = [
        w.lower() for w in texto 
        if w.lower() not in ingles
        and w not in string.punctuation
        and not any(c in w for c in ["|", "--", "''", "``", "()", "_", "-"])  # Verifica caracteres no deseados
    ]
    return texto_limpio
texto_limpio = quitarStopwords_eng(texto_tokenizado)

#4)Defino la función para obtener el pos de una palabra(separamos la palabra en verbo adjetivo o sustantivo)
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
#Defino la función para lematizar el texto
def lematizar(texto):
    texto_lema = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in texto]
    return texto_lema
#Inicializar el Lematizador
lemmatizer = WordNetLemmatizer()

#Printeo el texto tokenizado
print("Texto tokenizado:", texto_tokenizado)
print("-"*70)
#Printeo el texto sin las palabras vacias
print("Texto sin stopwords:", texto_limpio)
#Printeo el texto lematizado
print("-"*70)
print("Texto lematizado:", lematizar(texto_limpio))

#Aca preparo el corpus para tener todas las oraciones tokenizadas y separadas
corpus = [lematizar(quitarStopwords_eng(tokenizar(oracion))) for oracion in oraciones]
corpus_final = [" ".join(oracion) for oracion in corpus]
print ("Oraciones a analizar del corpus : ",corpus_final)

#Caso para convertir las palabras a numero
#Tf - idf
vectorizer = TfidfVectorizer()
#De corpus use las oraciones del txt
X = vectorizer.fit_transform(corpus_final)
#aca printeo la matriz
print("Matriz TF-IDF:")
print(X.toarray())
#aca printeo el vocabulario
print("\nVocabulario:")
print(vectorizer.get_feature_names_out())

# Obtengo las 6 palabras más usadas en todo el corpus
frecuencia = FreqDist(lematizar(quitarStopwords_eng(texto_limpio)))
palabras_mas_usadas = frecuencia.most_common(6)
print("Las 6 palabras mas usadas fueron : ",palabras_mas_usadas)
# Obtengo la palabra menos utilizada
palabra_menos_usada = min(frecuencia, key=frecuencia.get)
# Imprimir la palabra menos utilizada
print(f"\nLa palabra menos utilizada es: '{palabra_menos_usada}'")


# Analizar palabras más repetidas en cada oración
for i, oracion in enumerate(oraciones):
    # Tokenizar y procesar cada oración
    tokens_oracion = lematizar(quitarStopwords_eng(word_tokenize(oracion)))
    frecuencia_oracion = FreqDist(tokens_oracion)
    
    # Obtener la palabra más repetida en la oración
    palabra_mas_repetida = frecuencia_oracion.most_common(1)
    
    # Imprimir el resultado
    if palabra_mas_repetida:
        print(f"\nEn la oración {i + 1}, la palabra más repetida es: '{palabra_mas_repetida[0][0]}' con una frecuencia de {palabra_mas_repetida[0][1]}") 
    
    
#Grafico sin quitar las stopwords
frecuencia=FreqDist(texto_tokenizado)
frecuencia.plot(20, show=True)
#Grafico quitando las stopwords y lematizado
frecuencia=FreqDist(lematizar(quitarStopwords_eng(texto_limpio)))
frecuencia.plot(20, show=True)
