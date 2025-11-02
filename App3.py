import streamlit as st
import seaborn as sns
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURACIÓN DE LA PÁGINA
# =========================================================
st.set_page_config(
    page_title="Gastos en Salud",
    page_icon="💊",
    layout="wide"
)

# =========================================================
# TÍTULO Y DESCRIPCIÓN
# =========================================================
st.title("💊 Análisis interactivo de gastos en salud")
st.markdown("""
    Este conjunto de datos presenta los **gastos en salud** y la **esperanza de vida** 
    en distintos países entre **1970 y 2020**.  
    Permite comprender cómo la inversión en salud se relaciona con el bienestar y longevidad
    de las poblaciones, así como observar tendencias históricas y diferencias regionales.
""")

# =========================================================
# CARGA DE DATOS
# =========================================================
df = sns.load_dataset("healthexp")

# Verificar nombres de columnas
df.columns = df.columns.str.strip()

# =========================================================
# DESCRIPCIÓN DE VARIABLES
# =========================================================
st.markdown("## 📘 Descripción de las variables")
descripcion = {
    "Year": "El año de la observación, desde 1970 hasta 2020.",
    "Country": "El país de la observación (incluye Alemania, Francia, Reino Unido, Japón y EE. UU., entre otros).",
    "Spending_USD": "El gasto per cápita en salud, medido en dólares estadounidenses.",
    "Life_Expectancy": "La esperanza de vida promedio en años para ese país y año."
}
st.table(pd.DataFrame(list(descripcion.items()), columns=["Variable", "Descripción"]))


# =========================================================
# =========================================================
# DESCRIPCIÓN DE VARIABLES (botón en sidebar con toggle)
# =========================================================

st.sidebar.header("📘 Descripción de las variables")
mostrar_descripcion = st.sidebar.toggle("📖 Mostrar / Ocultar descripción de las variables", value=False)

# Mostrar u ocultar contenido según el estado del toggle
if mostrar_descripcion:
    st.markdown("""
    ---
    ### 🕓 **Year**
    Representa el **año de la observación**, abarcando un rango de **1970 a 2020**.  
    Es clave para analizar la **evolución temporal** del gasto sanitario y la esperanza de vida, 
    permitiendo identificar el impacto de políticas públicas, avances médicos y crisis económicas o sanitarias.

    ---

    ### 🌍 **Country**
    Identifica el **país de la observación**.  
    Incluye economías desarrolladas como **Alemania, Francia, Reino Unido, Japón y Estados Unidos**, 
    entre otras.  
    Esta variable es esencial para realizar **comparaciones internacionales**, observar diferencias
    estructurales en los sistemas de salud y analizar su relación con el desarrollo económico.

    ---

    ### 💰 **Spending_USD**
    Indica el **gasto per cápita en salud** (en **dólares estadounidenses**).  
    Representa la cantidad promedio invertida por persona en servicios médicos, hospitales,
    medicamentos y prevención.  
    Este valor incluye tanto gasto público como privado, siendo un **indicador directo del esfuerzo financiero**
    de cada país por mantener o mejorar la calidad de su sistema sanitario.

    ---

    ### ❤️ **Life_Expectancy**
    Corresponde a la **esperanza de vida promedio (en años)** en cada país y año.  
    Este indicador sintetiza múltiples factores: calidad de atención médica, acceso a servicios, 
    alimentación, educación y nivel de vida.  
    Al compararlo con el gasto en salud, se puede evaluar si la inversión efectivamente se traduce en 
    **mejoras reales en la longevidad y salud pública**.

    ---

    #### 💡 **Interpretación inicial:**  
    - Un aumento en *Spending_USD* junto con un incremento en *Life_Expectancy* sugiere una correlación positiva.  
    - Diferencias entre países indican cómo factores sociales, económicos y tecnológicos influyen 
      en la eficiencia del gasto sanitario.  
    - Analizar la evolución por *Year* permite identificar periodos de crecimiento o crisis 
      que afectaron los indicadores de salud global.
    """)
else:
    st.info("📘 Puedes ver la descripción completa de las variables haciendo clic en el botón del **panel lateral izquierdo 📖 (Mostrar / Ocultar descripción de las variables.)**")

# =========================================================
# PREVISUALIZACIÓN DE DATOS
# =========================================================
st.divider()
st.subheader("📂 Vista general de los datos")
st.dataframe(df, use_container_width=True)

# =========================================================
# VISUALIZACIÓN GENERAL (con botón de control)
# =========================================================
st.sidebar.header("📂 Vista general de los datos")
if "mostrar_graficos" not in st.session_state:
    st.session_state.mostrar_graficos = True

if st.sidebar.toggle("📚 Mostrar / Ocultar gráficos generales", value=True):
    st.subheader("📚 Visualización general de los datos")
    col1, col2 = st.columns(2)

    # --- Gráfico 1 ---
    with col1:
        st.write("### Distribución del gasto sanitario")
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="Country", y="Spending_USD", hue="Year", palette="Blues", ax=ax)
        ax.set_title("Gasto per cápita en salud por país y año")
        ax.set_xlabel("País")
        ax.set_ylabel("USD")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- Gráfico 2 ---
    with col2:
        st.write("### Relación entre gasto y esperanza de vida")
        fig2, ax2 = plt.subplots()
        sns.lineplot(
            data=df,
            x="Spending_USD",
            y="Life_Expectancy",
            hue="Country",
            marker="o",
            markersize=3.5,
            ax=ax2
        )
        ax2.set_title("Tendencia entre gasto y esperanza de vida")
        ax2.set_xlabel("Gasto en salud (USD)")
        ax2.set_ylabel("Esperanza de vida (años)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig2)

    # --- Segunda fila de gráficos ---
    col3, col4 = st.columns(2)

   # --- Gráfico 3 ---
    with col3:
        st.write("### Promedio de esperanza de vida por país")

        # Calcular promedios por país y total global
        avg_life_by_country = df.groupby("Country")["Life_Expectancy"].mean().reset_index()
        avg_total = df["Life_Expectancy"].mean()

        # Agregar fila del promedio total
        avg_life_by_country = pd.concat([
            avg_life_by_country,
            pd.DataFrame({"Country": ["Promedio Global"], "Life_Expectancy": [avg_total]})
        ], ignore_index=True)

        # Crear gráfico de barras
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        bars = sns.barplot(
            data=avg_life_by_country,
            x="Country",
            y="Life_Expectancy",
            palette="viridis",
            ax=ax3
        )

        # Títulos y etiquetas
        ax3.set_title("Esperanza de vida promedio por país y global (1970–2020)", fontsize=13, weight="bold")
        ax3.set_xlabel("País", fontsize=11)
        ax3.set_ylabel("Esperanza de vida promedio (años)", fontsize=11)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")

        # Añadir etiquetas de valor encima de cada barra
        for container in ax3.containers:
            ax3.bar_label(container, fmt="%.1f", padding=3, fontsize=10, weight="bold")

        # Destacar el promedio global con otro color o borde
        for patch, label in zip(ax3.patches, avg_life_by_country["Country"]):
            if label == "Promedio Global":
                patch.set_facecolor("#F4B400")  # color dorado para resaltar
                patch.set_edgecolor("black")
                patch.set_linewidth(1.2)

        # Líneas de referencia y estilo
        ax3.grid(axis="y", linestyle="--", alpha=0.6)
        ax3.set_ylim(0, avg_life_by_country["Life_Expectancy"].max() * 1.15)

        st.pyplot(fig3)



    # --- Gráfico 4 ---
    with col4:
        st.write("### Promedio de gasto en salud por país")
        fig4, ax4 = plt.subplots()
        gasto_promedio = df.groupby("Country")["Spending_USD"].mean().reset_index()
        sns.barplot(data=gasto_promedio, y="Country", x="Spending_USD", palette="Purples", ax=ax4)
        ax4.set_title("Promedio del gasto sanitario (USD)")
        ax4.set_xlabel("USD Promedio")
        ax4.set_ylabel("País")
        st.pyplot(fig4)

    # --- Descripción general ---
    st.markdown("""
    📈 Un aumento en *Spending_USD* junto con un incremento en *Life_Expectancy* sugiere una **correlación positiva**.  
    Las diferencias entre países reflejan cómo los factores **sociales, económicos y tecnológicos** influyen en la eficiencia del gasto sanitario.  
    Además, la evolución temporal permite identificar **tendencias sostenidas de mejora o estancamiento** en la salud pública.
    """)

else:
    st.info("""
    ℹ️ **Visualización deshabilitada**  
    Los gráficos generales están actualmente ocultos. Puedes activarlos desde el **panel lateral izquierdo** haciendo clic en  
    **📚 (Mostrar / Ocultar gráficos generales)**.
    """)

# =========================================================
# MÉTRICAS DESTACADAS
# =========================================================
st.divider()
st.subheader("🏆 Indicadores destacados")

col1, col2, col3, col4 = st.columns(4)

max_gasto = df.loc[df["Spending_USD"].idxmax()]
min_gasto = df.loc[df["Spending_USD"].idxmin()]
max_vida = df.loc[df["Life_Expectancy"].idxmax()]
min_vida = df.loc[df["Life_Expectancy"].idxmin()]

with col1:
    st.info(f"💰 **Mayor gasto:** {max_gasto['Country']}  \n {max_gasto['Spending_USD']} USD ({int(max_gasto['Year'])})")

with col2:
    st.success(f"💵 **Menor gasto:** {min_gasto['Country']}  \n {min_gasto['Spending_USD']} USD ({int(min_gasto['Year'])})")

with col3:
    st.warning(f"💕 **Mayor esperanza de vida:** {max_vida['Country']}  \n {max_vida['Life_Expectancy']} años")

with col4:
    st.error(f"⚰️ **Menor esperanza de vida:** {min_vida['Country']}  \n {min_vida['Life_Expectancy']} años")

st.divider()


# =========================================================
# ANÁLISIS DETALLADO CON TABS
# =========================================================
st.divider()
st.subheader("🔎 Análisis detallado")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Comparativo general",
    "📂Datos filtrados por pais",
    "🧩 Matriz de correlación",
    "📈 Tendencia global",
    "⚖️ Eficiencia sanitaria"
])

# =========================================================
# TAB 1 - COMPARATIVO GENERAL
# =========================================================
with tab1:
    st.write("#### Evolución temporal de la esperanza de vida")
    st.markdown("""
    Este gráfico muestra **cómo ha cambiado la esperanza de vida a lo largo del tiempo** en los distintos países del conjunto de datos. Permite identificar **tendencias generales de mejora**, así como **diferencias regionales** en el ritmo de crecimiento entre 1970 y 2020.
    """)

    fig3, ax3 = plt.subplots()
    sns.lineplot(data=df, x="Year", y="Life_Expectancy", hue="Country", marker="o", markersize=3.5, ax=ax3)
    ax3.set_title("Evolución de la esperanza de vida por país (1970–2020)")
    ax3.set_xlabel("Año")
    ax3.set_ylabel("Esperanza de vida (años)")
    ax3.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig3)

with tab2:
    st.markdown(" En esta sección puedes **explorar los datos específicos de cada país**. incluyendo los valores de gasto en salud y esperanza de vida registrados a lo largo del tiempo.")

    pais = st.selectbox("Selecciona un país del menú desplegable para visualizar su información:", df["Country"].unique())
    df_filtrado = df[df["Country"] == pais]
    st.dataframe(df_filtrado, use_container_width=True)

# =========================================================
# TAB 3 - MATRIZ DE CORRELACIÓN
# =========================================================
with tab3:
    st.subheader("🧩 Matriz de correlación")
    st.markdown("""
    Este mapa muestra la **relación estadística entre las variables numéricas** del conjunto de datos.  
    Un valor cercano a `1` indica correlación positiva fuerte (aumentan juntas), mientras que uno cercano a `-1` indica relación inversa.
    """)
    corr = df.select_dtypes("number").corr()

    fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax_corr)
    ax_corr.set_title("Correlación entre variables numéricas")
    st.pyplot(fig_corr)

    
# =========================================================
# TAB 4 - TENDENCIA GLOBAL
# =========================================================
with tab4:
    st.subheader("📈 Tendencia global a lo largo del tiempo")
    st.markdown("Este gráfico evidencia cómo el **incremento del gasto sanitario global** se ha acompañado de un **aumento sostenido de la esperanza de vida** en las últimas décadas.")

    df_trend = df.groupby("Year")[["Spending_USD", "Life_Expectancy"]].mean().reset_index()

    fig_trend = px.line(
        df_trend,
        x="Year",
        y=["Spending_USD", "Life_Expectancy"],
        markers=True,
        title="Evolución global del gasto y la esperanza de vida (1970–2020)"
    )

    fig_trend.update_layout(
        xaxis_title="Año",
        yaxis_title="Valor Promedio",
        template="plotly_white",
        legend_title_text="Variable"
    )

    st.plotly_chart(fig_trend, use_container_width=True)
    

# =========================================================
# TAB 5 - EFICIENCIA SANITARIA
# =========================================================
with tab5:
    st.subheader("⚖️ Análisis de eficiencia en gasto sanitario")
    st.markdown("""
    💡 Este análisis permite identificar qué países **logran una mayor esperanza de vida con menor gasto relativo**, 
    evidenciando una **mayor eficiencia** en el uso de sus recursos sanitarios.
    """)

    # Crear dataframe con promedios por país
    df_eff = df.groupby("Country", as_index=False)[["Spending_USD", "Life_Expectancy"]].mean()

    # Calcular recta de tendencia (y = m*x + b)
    import numpy as np
    m, b = np.polyfit(df_eff["Spending_USD"], df_eff["Life_Expectancy"], 1)

    # Crear gráfico de dispersión con línea de tendencia
    fig_eff = px.scatter(
        df_eff,
        x="Spending_USD",
        y="Life_Expectancy",
        text="Country",
        title="Relación entre gasto promedio y esperanza de vida (por país)"
    )

    fig_eff.add_scatter(
        x=df_eff["Spending_USD"],
        y=m * df_eff["Spending_USD"] + b,
        mode="lines",
        name="Tendencia (OLS)",
        line=dict(color="red", dash="dash")
    )

    fig_eff.update_traces(textposition="top center")
    fig_eff.update_layout(
        xaxis_title="Gasto promedio en salud (USD)",
        yaxis_title="Esperanza de vida promedio (años)",
        template="plotly_white"
    )

    st.plotly_chart(fig_eff, use_container_width=True)

    # Calcular eficiencia
    df_eff["Eficiencia"] = df_eff["Life_Expectancy"] / df_eff["Spending_USD"] * 1000

    st.markdown("#### 🧮 Eficiencia (años de vida por cada 1000 USD invertidos)")
    st.dataframe(df_eff.sort_values("Eficiencia", ascending=False), use_container_width=True)

    
st.markdown("<br><br>", unsafe_allow_html=True)

# =========================================================
# SIDEBAR DE FILTROS Y OPCIONES
# =========================================================
st.sidebar.header("💎 Anexo")
st.sidebar.header("🔎 Filtros y Visualización Dinámica")


st.markdown("""
### 💎 Anexo Interactivo: Filtros y Visualización Dinámica

Este anexo complementa el análisis principal mediante una **herramienta interactiva** desarrollada en *Streamlit*,  
diseñada para explorar los datos de forma flexible y visual.  

En la **barra lateral** se ubican los controles de filtrado, que permiten seleccionar **países, rangos de años, niveles de gasto sanitario y esperanza de vida**.  
En el **panel central**, el usuario puede elegir **las variables numéricas para los ejes X e Y**, generando así **gráficos personalizados** que facilitan el análisis comparativo entre distintas dimensiones del conjunto de datos.  

Esta funcionalidad interactiva amplía la interpretación del estudio, permitiendo descubrir relaciones, patrones y comportamientos específicos entre las variables analizadas.
       
""")

# --- Filtro por país ---
paises = sorted(df["Country"].unique())
# Seleccionar valores válidos automáticamente
default_paises = paises[:2] if len(paises) >= 2 else paises
paises_sel = st.sidebar.multiselect(
    "Selecciona país(es):",
    options=paises,
    default=default_paises
)

# --- Filtro por rango de años ---
year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
rango_anios = st.sidebar.slider(
    "Rango de años:",
    year_min, year_max, (year_min, year_max)
)

# --- Filtro por gasto ---
gasto_min, gasto_max = int(df["Spending_USD"].min()), int(df["Spending_USD"].max())
rango_gasto = st.sidebar.slider(
    "Rango de gasto (USD):",
    gasto_min, gasto_max, (gasto_min, gasto_max)
)

# --- Filtro por esperanza de vida ---
vida_min, vida_max = round(df["Life_Expectancy"].min(), 1), round(df["Life_Expectancy"].max(), 1)
rango_vida = st.sidebar.slider(
    "Rango de esperanza de vida (años):",
    float(vida_min), float(vida_max), (float(vida_min), float(vida_max))
)

# =========================================================
# APLICAR FILTROS
# =========================================================
df_filtrado = df[
    (df["Country"].isin(paises_sel)) &
    (df["Year"].between(rango_anios[0], rango_anios[1])) &
    (df["Spending_USD"].between(rango_gasto[0], rango_gasto[1])) &
    (df["Life_Expectancy"].between(rango_vida[0], rango_vida[1]))
]



mostrar_tabla = st.sidebar.checkbox("Mostrar tabla filtrada", value=False)
mostrar_grafico = st.sidebar.checkbox("Mostrar gráficos interactivos", value=False)


# =========================================================
# TABLA INTERACTIVA
# =========================================================

if mostrar_tabla:
    st.markdown("### 📋 Datos filtrados")
    st.markdown("En primer lugar, se genera una vista dinámica de los datos, construida a partir de los criterios de filtrado seleccionados por el usuario.")
    if not df_filtrado.empty:
        st.dataframe(df_filtrado, use_container_width=True)
    else:
        st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados.")
else:
    st.info("💡 Activa la opción en la barra lateral para visualizar la tabla filtrada.")

# =========================================================
# GRÁFICO INTERACTIVO PERSONALIZABLE
# =========================================================
if mostrar_grafico:
    if not df_filtrado.empty:
        st.markdown("### 📈 Visualización interactiva personalizada")

        # Selección de variables numéricas disponibles
        columnas_numericas = df_filtrado.select_dtypes(include=['number']).columns.tolist()

        if len(columnas_numericas) >= 2:
            # Selección de variables para los ejes
            col1, col2 = st.columns(2)
            with col1:
                eje_x = st.selectbox("Selecciona el eje X:", columnas_numericas, key="eje_x")

            opciones_y = [col for col in columnas_numericas if col != eje_x]
            with col2:
                eje_y = st.selectbox("Selecciona el eje Y:", opciones_y, key="eje_y")

            # Generar gráfico lineal dinámico
            fig = px.line(
                df_filtrado,
                x=eje_x,
                y=eje_y,
                color="Country" if "Country" in df_filtrado.columns else None,
                markers=True,
                title=f"Evolución de {eje_y} en función de {eje_x}",
            )

            fig.update_layout(
                legend_title_text="País" if "Country" in df_filtrado.columns else "",
                xaxis_title=eje_x.replace("_", " "),
                yaxis_title=eje_y.replace("_", " "),
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No hay suficientes variables numéricas para generar un gráfico.")
    else:
        st.warning("⚠️ No hay datos disponibles para generar gráficos.")
else:
    st.info("💡 Activa la opción en la barra lateral para visualizar los gráficos interactivos.")


    


st.caption("Elaborado por: **Juan David** — Trabajo Final de la Materia de Análisis de Datos con Python 🧠")