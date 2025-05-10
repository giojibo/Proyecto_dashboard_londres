
# Importamos librerías necesarias
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from funpymodeling.exploratory import freq_tbl 

# Definimos una paleta de colores personalizada
COLORS = {
    "primary": "#3a1de1",
    "secondary": "#4f2de9",
    "tertiary": "#653df0",
    "quaternary": "#7a4df8",
    "quinary": "#905dff"
}

# Aplicamos la paleta de colores a los elementos
st.set_page_config(page_title="Análisis de Datos de Londres", layout="wide")
st.markdown(f"""
    <style>
        .sidebar .sidebar-content {{
            background-color: {COLORS['primary']};
        }}
        .sidebar .sidebar .sidebar-header {{
            color: white;
        }}
        .stButton>button {{
            background-color: {COLORS['secondary']};
            color: white;
        }}
        .stTextInput>div>div>input {{
            background-color: {COLORS['tertiary']};
        }}
        .stMarkdown {{
            color: {COLORS['quinary']};
        }}
        .stWrite {{
            color: white;
        }}
    </style>
""", unsafe_allow_html=True)

# Definimos la instancia 
@st.cache_resource

# Creamos la función para cargar el dataset
def load_data():
    df = pd.read_csv("Datos_limpios_londres.csv")

    # Seleccionamos las columnas tipo numéricas del dataset
    numeric_df = df.select_dtypes(include=['float', 'int'])
    numeric_cols = numeric_df.columns

    # Seleccionamos las columnas tipo texto del dataset
    text_df = df.select_dtypes(['object'])
    text_cols = text_df.columns

    return df, numeric_cols, text_cols, numeric_df 

# Cargo los datos obtenidos del dataset
df, numeric_cols, text_cols, numeric_df = load_data()

# Crear una barra de navegación horizontal con imágenes y texto alineados
st.markdown("""
    <div class="navbar">
        <img src="https://www.seekpng.com/png/full/957-9571167_airbnb-png.png" width="100" />
        <span class="title" style="padding: 400px; font-size: 30px;">Airbnb Londres</span>
        <img src="https://wallpapercave.com/wp/wp2278615.jpg" width="100" />
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #3a1de1;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 12px;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        align-items: center;
        padding: 8px 10px;
        gap: 30px;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.2);
        z-index: 999;
    }

    .footer span {
        white-space: nowrap;
    }

    .footer .correo {
        font-style: italic;
    }
    </style>

    <div class="footer">
        <span>Alumno: Giovani Jimenez Bonilla</span>
        <span>Matrícula: 202128781</span>
        <span class="correo">Correo: giovani.jimenez@alumno.buap.mx</span>
        <span>Materia: Inteligencia de Negocios</span>
        <span>Profesor: Alfredo García Suárez</span>
        <span>Fecha: 2025-05-09</span>
    </div>
""", unsafe_allow_html=True)


# Barra de navegación con radio buttons para elegir la vista
View = st.radio(
    label="Vistas:",
    options=["Inicio", "Extraer Datos", "Regresión Lineal Simple", "Regresión Lineal Multiple", "Regresión Logistica", "Extraccion de Caracteristicas"],
    index=0,  # Esta es la vista predeterminada (Inicio)
    horizontal=True  # Esto hace que las opciones se muestren horizontalmente
)

# Contenido de la página 1
if View == "Inicio": 
    st.title("Análisis de datos de Londres")
    st.header("Inicio")
    st.subheader("Descripción del dataset")
    st.write("Este dataset contiene información relevante sobre Londres, el cual será empleado para un análisis de datos para la materia de inteligencia de negocios.")
    st.write("En este dataset se realizarán la extracción de datos, limpieza de datos, regresión lineal, regresión lineal múltiple y regresión logística.")
    st.image("http://wallpapercave.com/wp/tKCaN8t.jpg", caption="Londres")

    # Generamos los encabezados para la barra lateral (sidebar)
    st.sidebar.title("Menú")
    st.sidebar.header("Barra lateral")
    st.sidebar.subheader("Panel de selección")

    # Widget 2: Checkbox
    # Checkbox para mostrar el dataset completo
    show_data = st.sidebar.checkbox("Mostrar dataset completo")
    if show_data:
        st.subheader("Dataset completo")
        st.write(df)
    
    
    show_numeric_cols = st.sidebar.checkbox("Mostrar columnas numéricas")
    if show_numeric_cols:
        st.write("Columnas numéricas del dataset:")
        st.write(numeric_cols)
    
    show_text_cols = st.sidebar.checkbox("Mostrar columnas de texto")
    if show_text_cols:
        st.write("Columnas de texto del dataset:")
        st.write(text_cols)

# CONTENIDO DE LA VISTA 2
elif View == "Extraer Datos":
    # Generamos los encabezados para el dashboard
    st.title("Análisis de datos de Londres")
    st.header("Inicio")
    st.subheader("Extracción de datos")
    st.write("En esta sección se realizará la extracción de datos del dataset para posteriormente realizar un análisis de datos. El dataset contiene información sobre propiedades en Londres, incluyendo el tipo de propiedad, el precio, la ubicación y otros detalles relevantes.")
    st.write("El objetivo de esta sección es permitir al usuario seleccionar las columnas que desea extraer del dataset para su posterior análisis.")  

    st.sidebar.title("Menú")
    st.sidebar.header("Barra lateral")
    st.sidebar.subheader("Panel de selección")
    # Multiselect para seleccionar las columnas a extraer
    selected_cols = st.sidebar.multiselect("Selecciona las columnas numéricas a extraer", options=numeric_cols)
    selected_text_cols = st.sidebar.multiselect("Selecciona las columnas de texto a extraer", options=text_cols)

    # Función para extraer los datos
    def extract_data(selected_cols):
        return df[selected_cols]

    def extract_text_data(selected_text_cols):
        return df[selected_text_cols]

    # Extraer los datos
    extracted_data = extract_data(selected_cols)
    extracted_text_data = extract_text_data(selected_text_cols)

    # Unir ambos conjuntos de datos
    combined_data = pd.concat([extracted_data, extracted_text_data], axis=1)

    # Botón para mostrar los datos extraídos
    show_extracted_data = st.sidebar.button("Mostrar datos extraídos")
    if show_extracted_data:
        st.write("Datos extraídos:")
        st.write(combined_data)

        st.write("Columnas extraídas:")
        st.write(combined_data.columns.tolist())

        # Guardar en session_state
        st.session_state["extracted_data"] = combined_data

        if "extracted_data" in st.session_state:
            # Descargar CSV
            csv = combined_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar datos extraídos",
                data=csv,
                file_name="datos_extraídos.csv",
                mime="text/csv"
            )
        else:
            st.warning("No hay datos extraídos aún.")

# CONTENIDO DE LA VISTA 3
elif View == "Regresión Lineal Simple":
    # Encabezados del dashboard
    st.title("Análisis de datos de Londres")
    st.header("Inicio")
    st.subheader("Regresión lineal simple")

    # Barra lateral
    st.sidebar.title("Menú")
    st.sidebar.header("Barra lateral")
    st.sidebar.subheader("Panel de selección")

    # Selección de variables
    st.sidebar.markdown("**Selecciona tu variable independiente (X):**")
    x_var = st.sidebar.selectbox("Variable independiente (X)", options=numeric_cols)

    st.sidebar.markdown("**Selecciona tu variable dependiente (y):**")
    y_var = st.sidebar.selectbox("Variable dependiente (y)", options=[col for col in numeric_cols if col != x_var])

    # Botón para ejecutar la regresión
    show_regression = st.sidebar.button("Realizar regresión lineal")

    if show_regression:
        if x_var and y_var:
            # Extraemos las variables del DataFrame
            X = df[[x_var]]  # Debe ser un DataFrame, no una Serie
            y = df[y_var]

            # Mostramos las variables seleccionadas
            st.write("Variable independiente (X):")
            st.dataframe(X.head(5))  # Mostramos las primeras filas para verificar

            st.write("Variable dependiente (y):")
            st.dataframe(y.head(5))  # Mostramos las primeras filas para verificar  

            # Crear la figura y el eje
            fig, ax = plt.subplots()

            # Generar el scatter plot con seaborn sobre el eje creado
            sns.scatterplot(x=X[x_var], y=y, ax=ax, color='blue')

            # Mostrar la figura en Streamlit de forma segura
            st.pyplot(fig)

            # Entrenamos el modelo
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()

            # Ajustamos el modelo a los datos
            model.fit(X, y)

            # Mostramos los coeficientes del modelo
            st.write("Coeficientes del modelo:")
            st.write(model.__dict__)

            # Evaluamos la eficiencia del modelo
            model.score(X, y)
            st.write("Eficiencia del modelo:")
            st.write(model.score(X, y))

            # Realizamos la predicción
            y_pred = model.predict(X)

            # Añadimos las predicciones al DataFrame
            df['Predicciones'] = y_pred


            # Mostramos los resultados
            st.write("Resultados de la regresión lineal:")
            st.write("Coeficiente de determinación:")
            coef_determ = model.score(X, y)
            st.write(coef_determ)
            st.write("Coeficiente de correlación:")
            coef_corr = np.sqrt(coef_determ)
            st.write(coef_corr)
            
            st.write("Predicciones:")
            st.write(df[['Predicciones']].head(5))

            # Crear la figura y el eje
            fig2, ax = plt.subplots()
            # Visualizamos los resultados
            sns.scatterplot(x=X[x_var], y=y, color='blue')
            sns.scatterplot(x=X[x_var], y=y_pred, color='red')
            plt.title("Regresión Lineal Simple")
            st.pyplot(fig2)
        else:
            st.warning("Debe seleccionar tanto la variable independiente como la dependiente.")

    show_matrix = st.sidebar.checkbox("Mostrar matriz de correlación")
    if show_matrix:
        st.subheader("Matriz de correlación")
        df_num = df.select_dtypes(include=[np.number])
        corr = df_num.corr()

        correlation_matrix = abs(corr)
        st.write(correlation_matrix)

        head_map = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Matriz de correlación")
        st.pyplot(head_map.figure)

## CONTENIDO DE LA VISTA 4
elif View == "Regresión Lineal Multiple":
    # Encabezados del dashboard
    st.title("Análisis de datos de Londres")
    st.header("Inicio")
    st.subheader("Regresión lineal múltiple")

    # Barra lateral
    st.sidebar.title("Menú")
    st.sidebar.header("Barra lateral")
    st.sidebar.subheader("Panel de selección")

    # Selección de variables
    st.sidebar.markdown("**Selecciona tus variables independientes (X):**")
    # Seleccionamos más de una variable independiente (X) de las variables numéricas
    x_vars = st.sidebar.multiselect("Variables independientes (X)", options=numeric_cols)

    st.sidebar.markdown("**Selecciona tu variable dependiente (y):**")
    # Selección de la variable dependiente (y)
    y_var = st.sidebar.selectbox("Variable dependiente (y)", options=[col for col in numeric_cols if col not in x_vars])

    # Botón para ejecutar la regresión
    show_regression = st.sidebar.button("Realizar regresión lineal múltiple")

    if show_regression:
        if x_vars and y_var:
            # Extraemos las variables del DataFrame
            X = df[x_vars]  # Varias variables independientes (X)
            y = df[y_var]   # Variable dependiente (y)

            # Mostramos las variables seleccionadas
            st.write("Variables independientes (X):")
            st.dataframe(X.head(5))  # Mostramos las primeras filas para verificar

            st.write("Variable dependiente (y):")
            st.dataframe(y.head(5))  # Mostramos las primeras filas para verificar  

            # Crear la figura y el eje
            fig, ax = plt.subplots()

            #Colores automáticos para el scatter plot
            palette = sns.color_palette(n_colors=len(x_vars))

            # Generar el scatter plot para cada par de variables (X, y)
            for i, var in enumerate(x_vars):
                sns.scatterplot(x=X[var], y=y, ax=ax, label=var, color=palette[i])

            # Añadir título y leyenda
            ax.set_title("Relación entre variables independientes y dependiente")
            ax.set_xlabel("Variables independiente")
            ax.set_ylabel(y_var)
            ax.legend(title="Variables seleccionadas")

            # Mostrar la figura en Streamlit de forma segura
            st.pyplot(fig)

            # Entrenamos el modelo de regresión lineal múltiple
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()

            # Ajustamos el modelo a los datos
            model.fit(X, y)

            # Mostramos los coeficientes del modelo
            st.write("Coeficientes del modelo:")
            st.write(model.coef_)

            # Evaluamos la eficiencia del modelo
            st.write("Eficiencia del modelo (R^2):")
            model_score = model.score(X, y)
            st.write(model_score)

            # Realizamos la predicción
            y_pred = model.predict(X)

            # Mostramos los resultados
            st.write("Resultados de la regresión lineal múltiple:")
            st.write("Coeficiente de determinación (R^2):")
            coef_determ = model_score
            st.write(coef_determ)
            st.write("Coeficiente de correlación:")
            coef_corr = np.sqrt(coef_determ)
            st.write(coef_corr)
            
            st.write("Predicción:")
            st.write(y_pred)

            # Visualizamos los resultados (gráfico de dispersión de valores reales vs predichos)
            fig3, ax2 = plt.subplots()
            var = x_vars[0]  # Selecciona la primera variable de la lista
            sns.scatterplot(x=X[var], y=y, color='blue', ax=ax2, label='Valores reales')
            sns.scatterplot(x=X[var], y=y_pred, color='red', ax=ax2, label='Valores predichos')

            plt.title("Regresión Lineal Múltiple (una variable mostrada)")
            plt.xlabel(var)
            plt.ylabel(y_var)
            plt.legend()
            st.pyplot(fig3)

        else:
            st.warning("Debe seleccionar tanto las variables independientes como la dependiente.")

    show_matrix = st.sidebar.checkbox("Mostrar matriz de correlación")
    if show_matrix:
        st.subheader("Matriz de correlación")
        df_num = df.select_dtypes(include=[np.number])
        corr = df_num.corr()

        correlation_matrix = abs(corr)
        st.write(correlation_matrix)

        head_map = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Matriz de correlación")
        st.pyplot(head_map.figure)

# VISTA 5: Regresión Logística
elif View == "Regresión Logistica":
    st.title("Análisis de datos de Londres")
    st.header("Inicio")
    st.subheader("Regresión logística")
    st.write(
        "En esta sección se aplicará regresión logística para predecir una variable categórica binaria "
        "a partir de variables independientes numéricas."
    )

    # Mostrar clases de las columnas categóricas
    st.markdown("### Variables categóricas y su cantidad de clases:")
    for col in text_cols:
        st.write(f"- **{col}**: {df[col].nunique()} clases")

    # Creamos una copia del DataFrame para no modificar el original
    df_transformado = df.copy()

    # Transformación personalizada de variables categóricas
    st.markdown("### Transformación de variables categóricas:")
    for col in text_cols:
        num_clases = df_transformado[col].nunique()
        total_registros = len(df_transformado[col])

        if num_clases > 50:
            mitad = total_registros // 2
            df_transformado[col] = [True if i < mitad else False for i in range(total_registros)]
            st.write(f"🔄 {col} → Dividida en dos mitades (True/False) por tener más de 50 clases.")
        
        elif df_transformado[col].value_counts().max() > 1000:
            clase_mas_frecuente = df_transformado[col].value_counts().idxmax()
            df_transformado[col] = df_transformado[col].apply(lambda x: True if x == clase_mas_frecuente else False)
            st.write(f"🔄 {col} → Convertida a binaria según clase más frecuente ('{clase_mas_frecuente}').")
        
        elif num_clases == 2:
            primera_clase = df_transformado[col].value_counts().idxmax()
            df_transformado[col] = df_transformado[col].apply(lambda x: True if x == primera_clase else False)
            st.write(f"🔄 {col} → Ya binaria. Convertida según clase '{primera_clase}'.")

    # Identificamos las columnas binarias transformadas (True/False)
    binary_cols = [col for col in df_transformado.columns if df_transformado[col].dropna().nunique() == 2 and df_transformado[col].dtype == 'bool']

    # Sidebar para selección de variables
    st.sidebar.title("Selección de variables")
    y_var = st.sidebar.selectbox("Selecciona tu variable dependiente (binaria)", options=binary_cols)
    x_vars = st.sidebar.multiselect("Selecciona tus variables independientes (numéricas)", options=list(numeric_cols))

    # Botón para ejecutar la regresión logística
    ejecutar_regresion = st.sidebar.button("Ejecutar regresión logística")

    if ejecutar_regresion:
        if y_var and x_vars:
            # Definir X e y desde df_transformado
            X = df_transformado[x_vars]
            y = df_transformado[y_var]

            st.write("### Conteo de clases en y (binaria):")
            st.write(y.value_counts())

            # Verificamos que y sea binaria válida
            if y.nunique() < 2:
                st.warning("La variable dependiente solo tiene una clase. No se puede entrenar el modelo.")
            else:
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score

                # Separar datos
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

                # Escalar
                escalar = StandardScaler()
                X_train = escalar.fit_transform(X_train)
                X_test = escalar.transform(X_test)

                # Entrenamiento del modelo
                modelo = LogisticRegression()
                modelo.fit(X_train, y_train)

                # Predicción
                y_pred = modelo.predict(X_test)

                # Resultados
                st.write("### Matriz de confusión:")
                st.write(confusion_matrix(y_test, y_pred))

                st.write("### Precisión positiva:")
                st.write(precision_score(y_test, y_pred, pos_label=True))

                st.write("### Precisión negativa:")
                st.write(precision_score(y_test, y_pred, pos_label=False))

                st.write("### Exactitud:")
                st.write(accuracy_score(y_test, y_pred))

                st.write("### Sensibilidad positiva:")
                st.write(recall_score(y_test, y_pred, pos_label=True))

                st.write("### Sensibilidad negativa:")
                st.write(recall_score(y_test, y_pred, pos_label=False))
        else:
            st.warning("Selecciona una variable dependiente binaria y al menos una variable numérica independiente.")
    
elif View == "Extraccion de Caracteristicas":
    st.title("Análisis de datos de Londres")
    st.header("Inicio")
    st.subheader("Extracción de características")
    st.write("En esta sección se realizará la extracción de características del dataset para posteriormente realizar un análisis de datos. El dataset contiene información sobre propiedades en Londres, incluyendo el tipo de propiedad, el precio, la ubicación y otros detalles relevantes.")
    st.write("El objetivo de esta sección es permitir al usuario seleccionar las columnas que desea extraer del dataset para su posterior análisis.")  

    # Menú lateral
    st.sidebar.title("Menú")
    st.sidebar.header("Barra lateral")
    st.sidebar.subheader("Panel de selección")

    # Selector para tipo de variable: categórica o numérica
    tipo_variable = st.sidebar.radio("Selecciona el tipo de variable que deseas analizar:", ["Categórica (object)", "Numérica"])
    if tipo_variable == "Categórica (object)":
        st.sidebar.markdown("**Variables tipo object:**")
        selected_col = st.sidebar.selectbox("Selecciona una variable", options=text_cols)

        show_analysis_selected = st.sidebar.checkbox("Mostrar análisis univariado de la variable seleccionada", key="checkbox_obj_1")

        if show_analysis_selected:
            st.write(f"### Análisis Univariado de la variable categórica seleccionada: {selected_col}")
            table = freq_tbl(df[selected_col])
            st.write(table)

            # Mostrar solo la frecuencia
            table2 = table.drop(columns=['percentage', 'cumulative_perc'])
            st.write("### Frecuencia de la variable seleccionada:")
            st.write(table2)

            table2 = table2.sort_values(by='frequency', ascending=False).head(5)
            last_value = table2.iloc[-1]['frequency']
            filtro = table2[table2['frequency'] >= last_value]
            filtro_index = filtro.set_index(selected_col)

            st.write("### Valores más relevantes de la variable seleccionada:")
            st.write(filtro_index)

            show_bar_chart = st.sidebar.checkbox("Mostrar gráficos", key="checkbox_obj_plot")
            if show_bar_chart:
                col1, col2 = st.columns(2)

                # Gráfico de barras
                with col1:
                    fig_bar_obj, ax_bar_obj = plt.subplots(figsize=(6, 4))
                    filtro_index.plot(kind='bar', ax=ax_bar_obj)
                    ax_bar_obj.set_title(f"Barras: {selected_col}")
                    ax_bar_obj.set_xlabel("Categorías")
                    ax_bar_obj.set_ylabel("Frecuencia")
                    st.pyplot(fig_bar_obj)

                # Gráfico de pastel
                with col2:
                    fig_pie_obj, ax_pie_obj = plt.subplots(figsize=(6, 4))
                    filtro_index["frequency"].plot(
                        kind="pie",
                        ax=ax_pie_obj,
                        shadow=True,
                        autopct="%.1f%%"
                    )
                    ax_pie_obj.set_ylabel("")
                    ax_pie_obj.set_title(f"Pastel: {selected_col}")
                    st.pyplot(fig_pie_obj)

                col3, col4 = st.columns(2)

                # Gráfico de dispersión
                with col3:
                    if len(numeric_cols) > 0:
                        selected_num_for_plot = st.sidebar.selectbox("Variable numérica para graficar", options=numeric_cols, key="obj_scatter_select")
                        df_temp = df[[selected_col, selected_num_for_plot]].copy()
                        df_temp["frecuencia"] = df_temp[selected_col].map(df[selected_col].value_counts())

                        fig_disp_obj, ax_disp_obj = plt.subplots(figsize=(6, 4))
                        sns.scatterplot(x=df_temp["frecuencia"], y=df_temp[selected_num_for_plot], ax=ax_disp_obj)
                        ax_disp_obj.set_title(f"Dispersión: {selected_num_for_plot}")
                        ax_disp_obj.set_xlabel("Frecuencia")
                        ax_disp_obj.set_ylabel(selected_num_for_plot)
                        st.pyplot(fig_disp_obj)
                    else:
                        st.warning("No hay variables numéricas disponibles para graficar la dispersión.")

                # Gráfico hexbin
                with col4:
                    if len(numeric_cols) > 0:
                        fig_hex_obj, ax_hex_obj = plt.subplots(figsize=(6, 4))
                        hb = ax_hex_obj.hexbin(
                            x=df_temp["frecuencia"],
                            y=df_temp[selected_num_for_plot],
                            gridsize=20,
                            cmap='Oranges'
                        )
                        ax_hex_obj.set_title(f"Hexbin: Frecuencia vs {selected_num_for_plot}")
                        ax_hex_obj.set_xlabel("Frecuencia")
                        ax_hex_obj.set_ylabel(selected_num_for_plot)
                        cb = fig_hex_obj.colorbar(hb, ax=ax_hex_obj)
                        cb.set_label('N° de observaciones')
                        st.pyplot(fig_hex_obj)
                    else:
                        st.warning("No hay variables numéricas disponibles para el gráfico hexagonal.")

                # Gráfico de área
                col5, _ = st.columns(2)
                with col5:
                    fig_area_obj, ax_area_obj = plt.subplots(figsize=(10, 4))
                    filtro_index.plot(kind='area', ax=ax_area_obj, alpha=0.5)
                    ax_area_obj.set_title(f"Área: {selected_col}")
                    ax_area_obj.set_xlabel("Categorías")
                    ax_area_obj.set_ylabel("Frecuencia")
                    st.pyplot(fig_area_obj)

    elif tipo_variable == "Numérica":
        st.sidebar.markdown("**Variables tipo numérico:**")
        selected_col_num = st.sidebar.selectbox("Selecciona una variable numérica", options=numeric_cols)

        col_data = pd.to_numeric(df[selected_col_num], errors='coerce').dropna()
        n = len(col_data)

        if n == 0:
            st.error("La variable seleccionada no contiene datos numéricos válidos.")
        else:
            show_analysis_selected_num = st.sidebar.checkbox("Mostrar análisis univariado de la variable seleccionada", key="checkbox_num_1")

            if show_analysis_selected_num:
                st.write(f"### Análisis Univariado de la variable numérica seleccionada: {selected_col_num}")
                st.write(col_data.describe())

                # Cálculos
                Min = col_data.min()
                Max = col_data.max()
                Limites = [Min, Max]
                R = Max - Min
                ni = 1 + 3.32 * np.log10(n)
                ni2 = round(ni)
                i = R / ni2
                intervalos = np.linspace(Min, Max, num=ni2 + 1)

                categorias = []
                for j in range(len(intervalos) - 1):
                    categorias.append(f"{round(intervalos[j], 2)} - {round(intervalos[j+1], 2)}")

                df['cat_' + selected_col_num] = pd.cut(x=col_data, bins=intervalos, labels=categorias)

                st.write("### Información de la variable seleccionada:")
                st.write("- Límites:", Limites)
                st.write("- R:", R)
                st.write(f"- Número de intervalos (ni): {ni} redondeado a: {ni2}")
                st.write("- Ancho del intervalo:", i)
                st.write("- Categorías:", categorias)
                st.write(df['cat_' + selected_col_num].value_counts())

                # Tabla de frecuencias
                table_num = freq_tbl(df['cat_' + selected_col_num])
                table_num2 = table_num.drop(columns=['percentage', 'cumulative_perc'])
                st.write("### Análisis univariado:")
                st.write(table_num)
                st.write("### Frecuencia de la variable seleccionada:")
                st.write(table_num2)

                # Filtro para los 5 más frecuentes
                table_num2 = table_num2.sort_values(by='frequency', ascending=False).head(5)
                last_value_num = table_num2.iloc[-1]['frequency']
                filtro_num = table_num2[table_num2['frequency'] >= last_value_num]
                filtro_num_index = filtro_num.set_index('cat_' + selected_col_num)

                st.write("### Valores más relevantes y ajuste del índice:")
                st.write(filtro_num_index)

                show_bar_chart_num = st.sidebar.checkbox("Mostrar gráficos", key="checkbox_num_plot")
                if show_bar_chart_num:
                    col1, col2 = st.columns(2)

                    # Gráfico de barras
                    with col1:
                        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                        filtro_num_index.plot(kind='bar', ax=ax_bar)
                        ax_bar.set_title(f"Barras: {selected_col_num}")
                        ax_bar.set_xlabel("Categorías")
                        ax_bar.set_ylabel("Frecuencia")
                        st.pyplot(fig_bar)

                    # Gráfico de pastel
                    with col2:
                        fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
                        filtro_num_index["frequency"].plot(
                            kind="pie",
                            ax=ax_pie,
                            shadow=True,
                            autopct="%.1f%%"
                        )
                        ax_pie.set_ylabel("")  # Ocultar etiqueta y
                        ax_pie.set_title(f"Pastel: {selected_col_num}")
                        st.pyplot(fig_pie)

                    col3, col4 = st.columns(2)
                    # Grafico de dispersion
                    with col3:
                        fig_disp, ax_disp = plt.subplots(figsize=(6, 4))
                        sns.scatterplot(x=df['cat_' + selected_col_num], y=col_data, ax=ax_disp)
                        ax_disp.set_title(f"Dispersión: {selected_col_num}")
                        ax_disp.set_xlabel("Categorías")
                        ax_disp.set_ylabel(selected_col_num)
                        st.pyplot(fig_disp)
                    
                    #Grafico hexadecimal
                    with col4:
                        # Crear DataFrame temporal con frecuencia estimada
                        df_hex = df[['cat_' + selected_col_num]].copy()
                        df_hex["valor"] = col_data.values
                        df_hex["frecuencia"] = df_hex['cat_' + selected_col_num].map(df['cat_' + selected_col_num].value_counts())

                        fig_hex, ax_hex = plt.subplots(figsize=(6, 4))
                        hb = ax_hex.hexbin(
                            x=df_hex["frecuencia"],
                            y=df_hex["valor"],
                            gridsize=20,
                            cmap='Blues'
                        )
                        ax_hex.set_title(f"Hexbin: Frecuencia vs {selected_col_num}")
                        ax_hex.set_xlabel("Frecuencia")
                        ax_hex.set_ylabel(selected_col_num)
                        cb = fig_hex.colorbar(hb, ax=ax_hex)
                        cb.set_label('Número de observaciones')
                        st.pyplot(fig_hex)

                    col5, col6 = st.columns(2)

                    # Quinto gráfico: Área
                    with col5:
                        fig_area, ax_area = plt.subplots(figsize=(10, 4))
                        filtro_num_index.plot(kind='area', ax=ax_area, alpha=0.5)
                        ax_area.set_title(f"Área: {selected_col_num}")
                        ax_area.set_xlabel("Categorías")
                        ax_area.set_ylabel("Frecuencia")
                        st.pyplot(fig_area)
