import streamlit as st
import numpy as np
import pandas as pd
from modules.theme_toggle import initialize_theme, apply_styles
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from numpy.polynomial.polynomial import Polynomial
import pandasql as ps
import plotly.express as px
import plotly.graph_objects as go

# Inicializar el estado del tema
initialize_theme()

# Configuración de la página
st.set_page_config(page_title="DS - WOMPI", page_icon=":bar_chart:", layout="wide")

# Aplicar los estilos según el modo seleccionado
apply_styles()
st.markdown(
            """
            <style>
            .respuesta-box {
                background-color: #e6f4e6;
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
            }
            .respuesta-title {
                color: #4CAF50;
                font-weight: bold;
                font-size: 18px;
                margin-bottom: 10px;
            }
            </style>
            """, 
            unsafe_allow_html=True
        )
# URL del logo
logo_url = "https://wompi.com/assets/downloadble/logos_wompi/Wompi_LogoPrincipal.png"

# Variables para la selección de secciones
if 'selected_main' not in st.session_state:
    st.session_state.selected_main = "1"
    
# Barra lateral con el logo y menú
with st.sidebar:
    st.image(logo_url, use_column_width=True)  # Muestra el logo desde la URL
    st.title("Menú Principal")

     # Botones principales para secciones en filas de 5 columnas
    cols = st.columns(5)
    buttons = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    
    for i, button in enumerate(buttons):
        if cols[i % 5].button(button):
            st.session_state.selected_main = button

    
    # Botón para cambiar entre modo claro y oscuro con íconos de sol y luna
    theme_button = "🌙" if not st.session_state.dark_mode else "☀️"
    # st.button(theme_button, on_click=toggle_theme)

# Menú horizontal a la derecha basado en la selección
st.title("Prueba Técnica Proceso Científico de Datos")
st.subheader("Presentado por: Daniel Grass")



if st.session_state.selected_main:
    # Mostrar un menú horizontal según la selección del botón principal
    if st.session_state.selected_main == "1":
        
        # Generar datos aleatorios con ruido alrededor de una parábola
        np.random.seed(42)
        x = np.linspace(-10, 10, 100)
        y_real = 0.5 * x**2 - 3 * x + 2
        y_noise = y_real + np.random.normal(0, 5, x.shape[0])

        # Convertir a un DataFrame para facilitar el manejo
        data = pd.DataFrame({'x': x, 'y': y_noise})

        # Título de la aplicación
        st.title('1. Sesgo y Varianza')
        st.write("Describe en qué consiste el trade-off entre sesgo y varianza al entrenar un modelo. Y dibuja un ejemplo en un plano cartesiano en el que se ilustre un modelo con alto sesgo y otro con alta varianza.")
     
        

        # Contenido del cuadro de texto
        st.markdown(
             """
            <div class="respuesta-box">
                <div class="respuesta-title">Respuesta</div>
                En el entrenamiento de modelos de aprendizaje automático, el intercambio entre sesgo y varianza es un concepto clave. Se refiere al equilibrio entre dos fuentes de error que tienen un impacto en la capacidad del modelo para generalizar adecuadamente a datos nuevos:
                Cuando un modelo es demasiado simple para capturar las relaciones complejas en los datos, ocurre el <strong>sesgo</strong>. Un modelo con alto <strong>sesgo</strong> puede <strong>subajustar</strong>, lo que significa que hace predicciones inexactas y no aprende lo suficiente de los datos de entrenamiento. Por ejemplo, aplicar una regresión lineal a un conjunto de datos no lineales.
                La <strong>varianza</strong> es el error que ocurre cuando un modelo está demasiado ajustado a los datos de entrenamiento y es demasiado complejo. Un modelo que tiene una alta varianza puede <strong>sobreajustar</strong> (overfitting), lo que significa que se ajusta demasiado a las peculiaridades del conjunto de entrenamiento y falla al generalizar a nuevos datos.
                Encontrar un equilibrio adecuado entre el sesgo y la varianza es un desafío. El modelo ideal tiene un sesgo bajo y una varianza baja, lo que le permite un buen ajuste y generalización.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Gráfica 1: Regresión Lineal (Alto Sesgo)
        st.subheader('Gráfica 1: Alto Sesgo')
        lr = LinearRegression()
        lr.fit(x.reshape(-1, 1), y_noise)
        y_pred_lineal = lr.predict(x.reshape(-1, 1))

        # Crear la figura con plotly
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=x, y=y_noise, mode='markers', name='Datos'))
        fig1.add_trace(go.Scatter(x=x, y=y_pred_lineal, mode='lines', name='Regresión Lineal', line=dict(color='red')))
        st.plotly_chart(fig1)

        # Gráfica 2: Alta Varianza (Rectas que pasan por los puntos)
        st.subheader('Gráfica 2: Alta Varianza')
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=y_noise, mode='markers', name='Datos'))
        fig2.add_trace(go.Scatter(x=x, y=y_noise, mode='lines', name='Alta Varianza', line=dict(color='purple')))
        st.plotly_chart(fig2)

        # Gráfica 3: Ajuste Cuadrático (Bajo Sesgo y Baja Varianza)
        st.subheader('Gráfica 3: Bajo Sesgo y Baja Varianza')
        p = Polynomial.fit(x, y_noise, 2)
        y_pred_cuadratic = p(x)

        # Crear la figura con plotly
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=x, y=y_noise, mode='markers', name='Datos'))
        fig3.add_trace(go.Scatter(x=x, y=y_pred_cuadratic, mode='lines', name='Ajuste Cuadrático', line=dict(color='green')))
        st.plotly_chart(fig3)
    elif st.session_state.selected_main == "2":
         # Título de la aplicación
        st.title('2. Modelo de clasificación - F1 score')
        st.write("Estás entrenando un modelo de clasificación y obtienes un F1 score de 0.91 en el conjunto de entrenamiento y un F1 score de 0.18 en validación, ¿cuáles de las siguientes opciones utilizarías para resolver este problema?")
        st.write("a. Buscar variables nuevas que representen mejor el objetivo.")
        st.write("b. Boosting.")
        st.write("c. Disminuir la regularización.")
        st.write("d. Simplificar el modelo.")
        st.write("e. Regularizar el modelo.")

        # Contenido del cuadro de texto
        st.markdown(
             """
            <div class="respuesta-box">
                <div class="respuesta-title">Respuesta</div>
                Una de las formas más efectivas de combatir el sobreajuste es mediante la <strong>regularización del modelo (opción e)</strong>. La regularización (L1, L2 o una combinación) penaliza los parámetros del modelo para evitar que se ajusten demasiado a los datos de entrenamiento, lo que mejora su capacidad de generalización al reducir la varianza causada por la complejidad excesiva. Además, otra solución viable es <strong>simplificar el modelo (opción d)</strong>. Al reducir el número de parámetros, ya sea usando menos capas en una red neuronal o eliminando características redundantes, se puede mejorar aún más la capacidad de generalización, especialmente cuando el modelo es demasiado complejo.
                Buscar nuevas variables (opción a), aunque útil para mejorar el modelo, no resuelve directamente el sobreajuste, que es el problema principal aquí. El boosting (opción b), al agregar complejidad, podría empeorar el sobreajuste al hacer que el modelo se ajuste aún más a las particularidades del conjunto de entrenamiento. Por otro lado, disminuir la regularización (opción c)reduciría el control sobre la complejidad del modelo, lo que aumentaría aún más el ajuste a los datos de entrenamiento y empeoraría la capacidad de generalización. En este caso, lo más adecuado es simplificar y/o regularizar el modelo para reducir la varianza y mejorar la generalización.            </div>
            """,
            unsafe_allow_html=True
        )

    elif st.session_state.selected_main == "3":
        

        st.title("3. Modelo de Regresión - MSE")
        st.write("Te presentan los siguientes resultados de unos modelos usados para resolver el mismo problema de regresión:")

        # Crear los datos de la tabla en un DataFrame
        data = {
            'Algoritmo': ['Regresión Polinómica', 'Árbol de decisión', 'Máquinas de Soporte Vectorial', 'XGBoost'],
            'Error Cuadrático Medio (MSE)': [0.59, 0.78, 0.93, 0.57]
        }

        df = pd.DataFrame(data)

        # Mostrar la tabla
        st.write(df)

        # Pregunta a.
        st.write("### a. Argumenta cuál modelo seleccionarías para desplegar en producción")

        # Primer bloque de respuesta (equivalente a tcolorbox)
        st.markdown(
            """
            <div style="background-color: #e6f4e6; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px;">
            <strong>Respuesta:</strong> El modelo XGBoost tiene el menor MSE (0.57), lo que indica que sus predicciones son más cercanas a los valores reales en comparación con otros modelos, reflejando un mejor rendimiento en precisión. Aunque la regresión polinómica es más fácil de interpretar y tiene un MSE bajo (0.59), es importante considerar la complejidad del modelo. Para garantizar la consistencia de los resultados, es recomendable realizar validación cruzada o pruebas con datos independientes antes de tomar una decisión final. Para su implementación en producción, también se debe evaluar el impacto de la complejidad del modelo en los recursos computacionales.
            </div>
            """, 
            unsafe_allow_html=True
        )

        # Pregunta b.
        st.write("### b. ¿Te parece indicada la métrica de desempeño seleccionada? ¿Plantearías otra métrica?")

        # Segundo bloque de respuesta (equivalente a tcolorbox)
        st.markdown(
            """
            <div style="background-color: #e6f4e6; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px;">
            <strong>Respuesta:</strong> El Error Cuadrático Medio (MSE), la métrica elegida, es adecuada porque penaliza más severamente los errores grandes, lo que puede ser útil cuando esos errores son críticos. Pero combinarla con otras métricas puede dar una evaluación más completa. Por ejemplo, debido a que se encuentra en las mismas unidades que la variable objetivo, la Raíz del Error Cuadrático Medio (RMSE) es más fácil de interpretar. El Error Medio (ME) podría ser útil para determinar si el modelo tiene subajuste o sobreajuste. El Error Medio Absoluto (MAD) sería más robusto frente a valores atípicos. Finalmente, si queremos interpretar los errores en términos porcentuales, el Error Porcentual Absoluto Medio (MAPE) lo hace independiente de la escala de los datos y permite la comparación entre diferentes conjuntos de datos o modelos.
            </div>
            """, 
            unsafe_allow_html=True
        )
    elif st.session_state.selected_main == "4":
        st.title("4. Problema de clasificación binario")
        st.write("En un problema de clasificación binario entrenas un modelo cuya curva ROC es la siguiente:")

        # Mostrar la imagen ROC
        st.image('images/g4.png', caption='Curva ROC con AUC = 0.31',  width=300)

        # Descripción del problema
        st.write("""
        Un colega te dice que tendrías mejor resultado clasificando los datos lanzando una moneda, 
        (cara para positivo y sello para negativo). Argumenta si esto es cierto o no.
        """)

        # Cuadro con la justificación (simulando tcolorbox)
        st.markdown(
            """
            <div style="background-color: #e6f4e6; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px;">
            <strong>El colega tiene razón</strong> . La capacidad del modelo para distinguir entre clases se mide por el AUC (Área Bajo la Curva ROC), y un valor de 0.5 corresponde al rendimiento de un clasificador aleatorio, como lanzar una moneda (cara para positivo, sello para negativo). Esto se debe a que tienes una probabilidad del 50% de acertar cada predicción en un clasificador aleatorio.

            El desempeño del modelo es peor que el de un clasificador aleatorio, como se muestra en la gráfica, con una AUC de 0.31, un valor significativamente por debajo de 0.5. De hecho, un AUC tan bajo indica que el modelo está clasificando incorrectamente más veces de lo que lo haría al azar.
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.image('images/g5.png', caption='Curva ROC con AUC = 0.5',  width=300)
    elif st.session_state.selected_main == "5":
        st.title("5. Análisis de redes - fraude")
        st.write("""
        Tienes los siguientes atributos de una transacción de pago que realizó un consumidor en una tienda de ecommerce:
        """)

        # Lista de atributos de la transacción
        st.markdown("""
        - Id de la transacción.
        - Fecha y hora de la transacción.
        - Monto de la transacción.
        - Correo electrónico del pagador.
        - Teléfono del pagador.
        - Cédula del pagador.
        - Id de la tienda en la que se realizó la compra.
        - Id de la tarjeta de crédito con la que se realizó la compra.
        """)

        # Descripción del proyecto
        st.write("""
        Con esta información imagina que te asignan un proyecto en el que debes construir un grafo que te ayude a encontrar pagadores que puedan representar riesgo de fraude.
        """)

        # Subsección a.
        st.write("### a. Dibuja el esquema del grafo que construirías")

        # Cargar y mostrar imágenes del grafo
        st.image('images/g6.jpg')
        st.image('images/g7.jpg')
        st.image('images/g8.jpg')  
        # Subsección b.
        st.write("### b. ¿Cómo usarías este grafo para identificar usuarios con alto riesgo de fraude?")

        # Cuadro de texto con el análisis (simulando tcolorbox)
        st.markdown(
            """
            <div style="background-color: #e6f4e6; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px;">
            <strong>Análisis:</strong> El grafo puede detectar patrones de comportamiento inusuales. Podrías buscar, por ejemplo:
            <ul>
                <li><strong>Los pagadores están conectados a varias tarjetas:</strong> Si un pagador usa múltiples tarjetas en diferentes transacciones, esto puede indicar fraude.</li>
                <li><strong>Tarjetas de crédito compartidas entre pagadores:</strong> Si dos o más pagadores utilizan la misma tarjeta de crédito en diferentes tiendas, esto puede indicar actividad fraudulenta.</li>
                <li><strong>Tendencias en el tiempo:</strong> Utilizando las fechas y horas de las transacciones, podrías encontrar patrones como un número excesivamente alto de transacciones en un período de tiempo relativamente corto.</li>
            </ul>
            
            <p>Por ejemplo, para el usuario 1 y el usuario 2, podemos observar en el grafo que ambos están conectados a la misma tarjeta de crédito (CD 2) y participaron en las transacciones 3 y 4, respectivamente. Esta coincidencia en el uso de una tarjeta compartida puede ser un indicio de comportamiento sospechoso, ya que es inusual que diferentes usuarios legítimos compartan una tarjeta de crédito, a menos que sean parte del mismo núcleo familiar o se conozcan personalmente.</p>

            <p>Los algoritmos de detección de fraudes basados en grafos, como las redes neuronales en grafos (GNN) o el análisis de similitud, pueden ayudar a identificar patrones ocultos en los datos de comportamiento de los usuarios.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    elif st.session_state.selected_main == "6":
        st.title("6. SQL Transactions")
        st.write("""
        En una base de datos SQL tienes una tabla llamada `transactions`, la cual registra transacciones de pagos y cuenta con el siguiente esquema:
        """)

        # Crear una tabla en Streamlit usando markdown
        st.markdown("""
        <table border="1" cellspacing="0" cellpadding="5">
        <thead>
            <tr>
            <th>columna</th>
            <th>tipo</th>
            <th>descripción</th>
            </tr>
        </thead>
        <tbody>
            <tr>
            <td><code>id</code></td>
            <td><code>VARCHAR</code></td>
            <td>Identificador único de cada transacción</td>
            </tr>
            <tr>
            <td><code>created_at</code></td>
            <td><code>TIMESTAMP</code></td>
            <td>Momento de creación de la transacción</td>
            </tr>
            <tr>
            <td><code>payment_method_type</code></td>
            <td><code>VARCHAR</code></td>
            <td>Método de pago</td>
            </tr>
            <tr>
            <td><code>status_message</code></td>
            <td><code>VARCHAR</code></td>
            <td>Mensaje del estado de la transacción</td>
            </tr>
            <tr>
            <td><code>amount</code></td>
            <td><code>INT8</code></td>
            <td>Monto en pesos</td>
            </tr>
        </tbody>
        </table>
        """, unsafe_allow_html=True)

        # Explicación del query
        st.write("""
        Escribe un query en SQL que retorne la cantidad de transacciones por mensaje (`status_message`) y la participación porcentual de ese estado del total de transacciones, para el mes de agosto de 2024 y el medio de pago `'NEQUI'`.
        """)
        # Generar un DataFrame con al menos 500 registros
        np.random.seed(42)

        # Opciones para las columnas categóricas
        payment_method_type = ['NEQUI', 'Transferencia bancolombia', 'otra']
        status_message = ['invalida', 'confirmada', 'con_error', 'cancelada']

        # Generar 500 registros con datos aleatorios (timestamps para agosto de 2024)
        data = {
            'id': [f'TRX_{i+1}' for i in range(5000)],
            'created_at': np.random.randint(1707827200, 1730419200, 5000),  # Timestamps para agosto 2024
            'payment_method_type': np.random.choice(payment_method_type, size=5000),
            'status_message': np.random.choice(status_message, size=5000),
            'amount': np.random.randint(1000, 100000, size=5000)
        }
        st.markdown(
            """
            <div style="background-color: #e6f4e6; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px;">
            <strong>Generación de datos aleatorios:</strong>
            <p>En este ejemplo, se generarán <strong>5000 registros de transacciones</strong> de manera aleatoria utilizando la librería <code>numpy</code>. Para las transacciones, se generarán los siguientes campos:</p>
            <ul>
                <li><strong>ID de transacción</strong>: Identificador único para cada transacción, con el formato <code>TRX_1</code>, <code>TRX_2</code>, ..., <code>TRX_5000</code>.</li>
                <li><strong>Fecha y hora de la transacción (created_at)</strong>: Se generan timestamps en formato UNIX que corresponden a fechas del 2024.</li>
                <li><strong>Método de pago (payment_method_type)</strong>: Se seleccionará de manera aleatoria entre tres opciones: <code>'NEQUI'</code>, <code>'Transferencia bancolombia'</code>, y <code>'otra'</code>.</li>
                <li><strong>Estado de la transacción (status_message)</strong>: Se generará aleatoriamente uno de los siguientes valores: <code>'invalida'</code>, <code>'confirmada'</code>, <code>'con_error'</code>, <code>'cancelada'</code>.</li>
                <li><strong>Monto (amount)</strong>: Se generará un valor aleatorio en pesos entre $1,000 y $100,000.</li>
            </ul>
            <p>Este conjunto de datos simula las transacciones de un sistema de pagos para un análisis posterior.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        transactions = pd.DataFrame(data)
        st.write(transactions)
        st.markdown(
            """
            <div style="background-color: #e6f4e6; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px;">
            <strong>Se utilizará una consulta en SQL</strong> para obtener la <strong>cantidad de transacciones por estado (<code>status_message</code>)</strong> y su <strong>participación porcentual</strong> en relación con el total de transacciones del mes de agosto de 2024 y el método de pago <code>'NEQUI'</code>. La estructura de la consulta es la siguiente:
            <ol>
                <li><strong>Filtrado de datos</strong>: Filtramos solo las transacciones realizadas con el método de pago <code>'NEQUI'</code> y que corresponden a agosto de 2024.</li>
                <li><strong>Agrupación por estado de la transacción</strong>: Agrupamos las transacciones por el campo <code>status_message</code>, lo que nos permitirá contar cuántas transacciones se realizaron para cada estado.</li>
                <li><strong>Cálculo de participación porcentual</strong>: Calculamos el porcentaje que representa cada tipo de mensaje (<code>status_message</code>) en relación con el total de transacciones con el método <code>'NEQUI'</code>.</li>
            </ol>
            </div>
            """, 
            unsafe_allow_html=True
        )
        # Query para la cantidad de transacciones por mensaje (status_message) y la participación porcentual de ese estado del total de transacciones, para agosto de 2024 y el medio de pago 'NEQUI'
        query = """
            SELECT status_message, 
                COUNT(*) as cantidad_transacciones,
                (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions WHERE payment_method_type = 'NEQUI' AND strftime('%Y-%m', datetime(created_at, 'unixepoch')) = '2024-08')) as participacion_porcentual
            FROM transactions
            WHERE payment_method_type = 'NEQUI' 
                AND strftime('%Y-%m', datetime(created_at, 'unixepoch')) = '2024-08'
            GROUP BY status_message;
        """
        
        st.image('images/g9.png')  
        # Ejecutar la consulta con pandasql
        result = ps.sqldf(query, locals())

        st.write("Resultado consulta:")
        st.write(result)

    elif st.session_state.selected_main == "7":
        st.title("7. Spark")

        # Descripción inicial
        st.write("""
        Acabas de escribir un script en Spark para procesar un archivo de 8 millones de registros. Lo ejecutas en un laptop con 16GB de RAM y un procesador de última generación. Después lo ejecutas en un entorno distribuido de nodos de 3GB de RAM y procesadores sencillos (sumando los nodos tienen menos recursos de cómputo que tu equipo). El mismo script tiene un tiempo de ejecución menor en el entorno distribuido que en el entorno local, explica porque ocurre esto.
        """)

        # Cuadro de explicación con color de fondo
        st.markdown(
            """
            <div style="background-color: #e6f4e6; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px;">
            <strong>Spark divide los 8 millones de registros en múltiples particiones en un entorno distribuido y procesa cada partición en paralelo en varios nodos, lo que permite que el trabajo se realice más rápido, incluso si cada nodo tiene menos capacidad.</strong> Spark en un solo equipo puede ralentizar la ejecución debido a cuellos de botella de memoria y operaciones de I/O, a pesar de que su laptop tiene más RAM y un procesador más potente. Los nodos en un clúster distribuido gestionan mejor la memoria y el procesamiento paralelo mejora el rendimiento. Spark utiliza la distribución de trabajo, lo que explica por qué, a pesar de los recursos limitados de cada nodo, el tiempo de ejecución es menor en el entorno distribuido. 
            <a href='https://medium.com/@sephinreji98/understanding-spark-cluster-modes-client-vs-cluster-vs-local-d3c41ea96073' target='_blank'>Más información</a>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif st.session_state.selected_main == "8":
        st.title("8. Datalake en S3")

        # Descripción inicial
        st.write("""
        La nueva aplicación móvil genera registros de todas las interacciones de los usuarios. Vas a construir un modelo de segmentación de usuarios y el primer paso es consolidar estos registros en un data lake en S3. Puedes almacenar la información en formato CSV o Parquet y las estimaciones del tamaño en disco de cada formato para los registros de 1 mes son las siguientes:
        """)

        # Crear el DataFrame para la tabla de Formatos
        data_format = pd.DataFrame({
            "Formato": ["CSV", "Parquet"],
            "Tamaño en disco": ["1TB", "130GB"]
        })

        # Mostrar la tabla de Formatos
        st.write("### Formatos de almacenamiento")
        st.write(data_format)

        # Pregunta a.
        st.subheader("a. ¿Cuál formato eliges? Argumenta tu respuesta.")

        # Cuadro de respuesta
        st.markdown(
            """
            <div style="background-color: #e6f4e6; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px;">
            <strong>Elegiría el formato Parquet</strong> porque, al ser columnar, permite una mayor compresión y eficiencia de lectura para grandes volúmenes de datos, lo que reduce significativamente el espacio en disco (130GB frente a 1TB) y mejora el rendimiento en análisis.
            </div>
            """, 
            unsafe_allow_html=True
        )

        # Pregunta b.
        st.subheader("b. ¿Qué ventajas y desventajas encuentras en cada formato? Argumenta tu respuesta.")

        # Tabla de ventajas y desventajas
        st.markdown("""
        | **Formato** | **Ventajas**                                               | **Desventajas**                                                      |
        |-------------|-------------------------------------------------------------|----------------------------------------------------------------------|
        | CSV         | - Fácil de leer en cualquier editor de texto.               | - Tamaño de archivo grande.                                           |
        |             | - No requiere software especializado.                      | - Procesamiento lento en grandes volúmenes de datos.                  |
        | Parquet     | - Alta compresión y menor uso de disco.                    | - Requiere herramientas compatibles para lectura.                     |
        |             | - Eficiente para consultas analíticas.                      | - Más complejo para manipulación manual.                              |
        """)
    elif st.session_state.selected_main == "9":
        st.title("9. Anomalías Transacciones")

        # Descripción inicial
        st.write("""
        El pasado martes 17 de septiembre cursaron 188,756 transacciones por el sistema. La media de los 10 martes anteriores es de 191,050 transacciones. La mediana de los 10 martes anteriores es de 90,968 transacciones. Si comparo el 17 de septiembre contra la media el día parece normal, pero si lo comparo contra la mediana el tráfico es significativamente mayor. ¿Deberíamos preocuparnos por algun compartamiento extraño el 17 de septiembre? ¿qué recomendaciones me darías sobre esta medición?
        """)

        # Cuadro de explicación con color de fondo
        st.markdown(
            """
            <div style="background-color: #e6f4e6; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px;">
            <strong>A pesar de que el número de transacciones del 17 de septiembre parece ser similar al promedio de los 10 martes anteriores, la discrepancia con la mediana indica una distribución asimétrica de los datos.</strong> La media puede ser sesgada debido a valores únicos (posibles atípicos) o días de gran tráfico, lo que la hace menos representativa de un martes normal. En presencia de estos valores extremos, la mediana muestra mejor el comportamiento central. El hecho de que las transacciones del 17 de septiembre estén mucho por encima de la mediana podría ser una señal de un comportamiento inusual. Es crucial investigar si el aumento de tráfico se debió a eventos extraordinarios, promociones o problemas técnicos.

            Para monitorear el tráfico y detectar comportamientos anómalos, recomiendo utilizar modelos de pronóstico en series de tiempo univariadas como ARIMA, Prophet, modelos de suavizamiento exponencial o LSTM. Estos modelos permiten estimar un valor pronosticado y compararlo con el valor real. 

            Se define un intervalo de confianza con un límite superior e inferior, utilizando el error cuadrático medio (MSE) como base. Un factor de seguridad ajusta el ancho del intervalo. Si el valor real cae fuera del rango esperado, se considera una anomalía. Este enfoque proporciona una metodología robusta para detectar variaciones inusuales en el tráfico y prevenir comportamientos anómalos.
            </div>
            """, 
            unsafe_allow_html=True
        )

        # Mostrar las imágenes (asegúrate de tener las imágenes guardadas localmente)
        st.image("images/g12.png")
        st.image("images/g13.png")
        dates = pd.date_range(start='2020-04-01', end='2020-07-31', freq='D')
        y_hat = np.linspace(800, 1000, len(dates))  # Valores estimados
        mse = 10000
        s = 1.96  # Factor de seguridad para 95% de confianza
        ul = y_hat + s * np.sqrt(mse)  # Upper limit
        ll = y_hat - s * np.sqrt(mse)  # Lower limit
        y_real = y_hat + np.random.normal(0, 50, len(dates))  # Valores reales
        anomalies = [50, 60, 110, 120]  # Puntos de anomalías

        # Introducir anomalías en ciertos puntos
        y_real[anomalies] += np.random.normal(300, 20, len(anomalies))

        # Crear la figura
        fig = go.Figure()

        # Agregar línea de valor estimado (y_hat)
        fig.add_trace(go.Scatter(x=dates, y=y_hat, mode='lines', name='y_hat (estimated value)', line=dict(color='orange')))

        # Agregar límites superior (UL) e inferior (LL)
        fig.add_trace(go.Scatter(x=dates, y=ul, mode='lines', name='UL (Upper limit)', line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=dates, y=ll, mode='lines', name='LL (Lower limit)', line=dict(color='purple')))

        # Agregar valores reales (y) sin anomalías
        fig.add_trace(go.Scatter(x=dates, y=y_real, mode='markers', name='y (real value)', marker=dict(color='blue')))

        # Agregar anomalías
        fig.add_trace(go.Scatter(x=dates[anomalies], y=y_real[anomalies], mode='markers', name='Anomalies', marker=dict(color='red', size=10)))

        # Etiquetas y diseño
        fig.update_layout(
            title="Detección de Anomalías",
            xaxis_title="Date [2020]",
            yaxis_title="Valor",
            legend_title="Leyenda",
            showlegend=True
        )

        # Mostrar gráfico en Streamlit
        st.plotly_chart(fig)
    elif st.session_state.selected_main == "10":
        st.title("10. Ruleta de casino")

        st.write("""
        Imagina que estás jugando a la ruleta en un casino. La ruleta tiene 18 espacios rojos, 18 espacios negros, y 2 verdes (cero y doble cero). En las últimas 8 tiradas ha salido rojo cada vez. Tu vecino en la mesa piensa que ahora es más probable que salga negro en la siguiente tirada, dado que ha salido rojo tantas veces seguidas. ¿Qué opinas sobre las expectativas de tu vecino sobre de la próxima tirada?
        """)

        # Cuadro explicativo con color de fondo
        st.markdown(
            """
            <div style="background-color: #e6f4e6; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px;">
            <strong>Las expectativas de tu vecino se basan en la falacia del jugador</strong>, que es la creencia errónea de que procesos independientes influyen en las probabilidades de eventos pasados. Cada tirada de una ruleta es un evento independiente, lo que significa que la probabilidad de que salga rojo, negro o verde no cambia, sin importar lo que haya ocurrido antes.

            Las probabilidades de que salga un color en la siguiente tirada son las siguientes:
            <ul>
                <li><strong>Rojo:</strong> 18/38 </li>
                <li><strong>Negro:</strong> 18/38 </li>
                <li><strong>Verde (cero o doble cero): 2/38</li>
            </ul>
            La probabilidad de la próxima tirada no está influenciada por la secuencia de resultados anteriores (rojo en las últimas 8 tiradas), que sigue siendo la misma en cada tirada. Como cada tirada es independiente y no está influenciada por las tiradas anteriores, las <strong>expectativas de tu vecino son incorrectas</strong>.
            </div>
            """, 
            unsafe_allow_html=True
        )
    elif st.session_state.selected_main == "11":
        st.title("11. Análisis de Satisfacción")

        # Descripción inicial
        st.write("""
        Se presentan el histograma y el boxplot de las valoraciones de 1 a 10 proporcionadas por 1000 clientes sobre 3 servicios que ofrecemos (A, B y C). Una valoración alta quiere decir que el cliente tiene una satisfacción alta.
        """)

        # Mostrar imagen
        st.image("images/g14.png")
        # Pregunta a. Asocia e indica cuál es el boxplot de cada histograma.
        st.subheader("a. Asocia e indica cuál es el boxplot de cada histograma.")

        st.markdown(
            """
            <div style="background-color: #e6f4e6; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px;">
            <ul>
                <li><strong>A - 3:</strong> El histograma A muestra una distribución casi uniforme de las valoraciones de satisfacción, lo que coincide con el boxplot 3, que presenta un rango amplio y dispersión equitativa.</li>
                <li><strong>B - 1:</strong> El histograma B tiene una distribución simétrica con un pico en valores intermedios, lo cual coincide con el boxplot 1, que tiene una mediana central y presenta más dispersión en los extremos (outliers).</li>
                <li><strong>C - 2:</strong> El histograma C tiene una distribución sesgada hacia la izquierda con predominancia de valores bajos, lo cual coincide con el boxplot 2, que refleja esta misma tendencia con una mediana baja.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )       


        # Función para visualizar las distribuciones sin remover outliers
        def plot_distribution(column, df):
            # Visualización inicial de la distribución sin eliminar outliers
            fig_hist = px.histogram(df, x=column, nbins=50, title=f"Histograma de {column}", marginal="box")
            st.plotly_chart(fig_hist)

        # Generar datos de ejemplo
        np.random.seed(42)
        df = pd.DataFrame({
            'Servicio A': np.random.uniform(1, 10, 1000),
            'Servicio B': np.clip(np.random.normal(5, 1.5, 1000), 1, 10),
            'Servicio C': np.clip(np.random.exponential(1, 1000) * 3 + 1, 1, 10)
        })

        # Mostrar histogramas y boxplots para cada servicio sin eliminar outliers
        for column in df.columns:
            st.subheader(f"Distribución para {column}")
            plot_distribution(column, df)

        
        # Pregunta b. ¿En cuál servicio no hay una percepción clara de la satisfacción?
        st.subheader("b. ¿En cuál servicio no hay una percepción clara de la satisfacción?")

        st.markdown(
            """
            <div style="background-color: #e6f4e6; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px;">
            En el <strong>servicio A</strong>, la satisfacción no es clara. El histograma muestra una distribución más uniforme de las valoraciones, lo que indica que las opiniones están dispersas en casi todos los niveles, sin una tendencia clara hacia la satisfacción alta o baja.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Pregunta c. ¿En cuál servicio la satisfacción es muy baja?
        st.subheader("c. ¿En cuál servicio la satisfacción es muy baja?")

        st.markdown(
            """
            <div style="background-color: #e6f4e6; border-left: 5px solid #4CAF50; padding: 10px; border-radius: 5px;">
            En el <strong>servicio C</strong>, la satisfacción es baja. Tanto el histograma como el boxplot muestran que la mayoría de los clientes otorgan valoraciones bajas, ya que hay una clara acumulación de puntuaciones hacia el extremo inferior de la escala de satisfacción.
            </div>
            """,
            unsafe_allow_html=True
        )       
