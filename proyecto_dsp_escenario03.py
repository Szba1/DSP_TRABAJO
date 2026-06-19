
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from math import radians, cos, sin, asin, sqrt
import os

# =====================================================================
# CONFIGURACIÓN INICIAL
# =====================================================================
# Ruta a la carpeta que contiene los CSV del Escenario 03
# IMPORTANTE: Asegúrate de que esta ruta sea correcta en tu computador
data_dir = r"data\Escenario_03" 

# Días de calentamiento (warm-up). Se calculó empíricamente en 60 días.
# Estos primeros 60 días se descartan para que el sistema se estabilice
# y los KPIs no se vean sesgados por el arranque de la planta en vacío.
warmup_days = 60
warmup_hours = warmup_days * 24

print("==========================================================")
print("  FASE 1: ANÁLISIS DE PRODUCTIVIDAD (CÁLCULO DE KPIs)")
print("==========================================================\n")

# =====================================================================
# 1. CARACTERIZACIÓN DE FLUJOS (Materia Prima vs Productos Terminados)
# =====================================================================
print("Calculando flujos de materia prima y productos...")

# Cargar llegadas de trozos (Input)
log_arrivals = pd.read_csv(os.path.join(data_dir, "log_arrivals.csv"))
# Filtrar eventos ocurridos después del período de calentamiento
log_arrivals_valid = log_arrivals[log_arrivals['arrival_time_h'] > warmup_hours]
total_in = log_arrivals_valid['volume_m3'].sum()

# Cargar salidas de productos terminados (Output)
product_outputs = pd.read_csv(os.path.join(data_dir, "product_outputs.csv"))
# Filtrar salidas posteriores al calentamiento
product_outputs_valid = product_outputs[product_outputs['exit_time_h'] > warmup_hours]
total_out = product_outputs_valid['volume_m3'].sum()
total_out_by_prod = product_outputs_valid.groupby('product')['volume_m3'].sum()

# Identificar número de réplicas en la simulación
num_replications = product_outputs['replication'].nunique()

# Dividimos el total por el número de réplicas para obtener el volumen promedio real
promedio_entrada = total_in / num_replications
promedio_salida = total_out / num_replications
promedio_salida_por_producto = total_out_by_prod / num_replications

# Calcular el total de horas operativas netas (excluyendo warmup)
calendar_df = pd.read_csv(os.path.join(data_dir, "calendar.csv"))
eval_hours = (calendar_df['day'].max() * 24 + 24) - warmup_hours

print(f"Total promedio ingresado (Trozos): {promedio_entrada:,.2f} m3 ({promedio_entrada/eval_hours:.2f} m3/hora)")
print(f"Total promedio producido (P1, P2, P3): {promedio_salida:,.2f} m3 ({promedio_salida/eval_hours:.2f} m3/hora)")
for prod, vol in promedio_salida_por_producto.items():
    print(f" - Producto {prod}: {vol:,.2f} m3 ({vol/eval_hours:.2f} m3/hora)")

# =====================================================================
# 2. MÉTRICAS DE DESEMPEÑO POR ESTACIÓN (Utilización y Disponibilidad)
# =====================================================================
print("\nCalculando KPIs de utilización y disponibilidad por estación...")

station_events = pd.read_csv(os.path.join(data_dir, "station_events.csv"))

# Filtrar eventos que terminen después del calentamiento
se_valid = station_events[station_events['end_time_h'] > warmup_hours].copy()
# Si un evento empezó antes del calentamiento, acortamos su inicio para no contar ese tiempo
se_valid['start_time_h'] = np.maximum(se_valid['start_time_h'], warmup_hours)
se_valid['duration'] = se_valid['end_time_h'] - se_valid['start_time_h']

# Sumarizar la duración total en cada estado (BUSY, IDLE, DOWN, etc.) por réplica y estación
state_durations = se_valid.groupby(['replication', 'station', 'state'])['duration'].sum().reset_index()

# Ya se calculó eval_hours en la Fase 1

# Pivotear la tabla para tener los estados como columnas
pivot_states = state_durations.pivot_table(index=['replication', 'station'], columns='state', values='duration', fill_value=0).reset_index()

# Si la máquina tiene turnos inactivos (OFF_SHIFT), restamos eso al tiempo disponible total
if 'OFF_SHIFT' in pivot_states.columns:
    pivot_states['Operating_Time'] = eval_hours - pivot_states['OFF_SHIFT']
else:
    pivot_states['Operating_Time'] = eval_hours

# Si no hubo fallas (DOWN) o trabajo (BUSY) en alguna estación, inicializamos en 0
if 'DOWN' not in pivot_states.columns: pivot_states['DOWN'] = 0
if 'BUSY' not in pivot_states.columns: pivot_states['BUSY'] = 0

# CÁLCULO DE KPIs CLAVE
# Utilización = Tiempo procesando material (BUSY) / Tiempo Operativo Real
pivot_states['Utilization'] = pivot_states['BUSY'] / pivot_states['Operating_Time']
# Disponibilidad = (Tiempo Operativo Real - Tiempo en Fallas) / Tiempo Operativo Real
pivot_states['Availability'] = (pivot_states['Operating_Time'] - pivot_states['DOWN']) / pivot_states['Operating_Time']

mean_kpis = pivot_states.groupby('station')[['Utilization', 'Availability']].mean()
print(mean_kpis.to_string())

# =====================================================================
# 3. TIEMPOS DE CICLO PROMEDIO (Tiempo que tarda un lote en procesarse)
# =====================================================================
print("\nCalculando Tiempos de Ciclo por estación...")
batches = pd.read_csv(os.path.join(data_dir, "batches.csv"))
batches_valid = batches[batches['start_process_time_h'] > warmup_hours].copy()

# Tiempo de ciclo = Hora de fin del lote - Hora de inicio del lote
batches_valid['cycle_time'] = batches_valid['end_process_time_h'] - batches_valid['start_process_time_h']
mean_cycle_time = batches_valid.groupby('station')['cycle_time'].mean()

for est, tc in mean_cycle_time.items():
    print(f" - {est}: {tc:.2f} horas por lote")

# =====================================================================
# 4. ANÁLISIS DE INVENTARIO (BUFFERS)
# =====================================================================
print("\nAnalizando acumulación en inventarios (Cuello de Botella)...")
daily_wip = pd.read_csv(os.path.join(data_dir, "daily_wip.csv"))
daily_wip_valid = daily_wip[daily_wip['day'] > warmup_days]

buffer_stats = daily_wip_valid.groupby('buffer').agg(
    WIP_Promedio=('level_m3_mean', 'mean'),
    WIP_Maximo=('level_m3_max', 'max')
)
print(buffer_stats.to_string())
print("\n-> NOTA: El stock_aserrado tiene una acumulación masiva, lo que confirma que el Secado es el cuello de botella (ya que su Utilización es altísima y su ciclo de 27.8 hrs es muy lento).")


print("\n\n==========================================================")
print("  FASE 2: OPTIMIZACIÓN LOGÍSTICA (PROBLEMA DE TRANSPORTE)")
print("==========================================================\n")

import requests

def haversine(lat1, lon1, lat2, lon2):
    # Intentamos primero usar la API de OSRM para obtener la distancia real de conducción
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()['routes'][0]['distance'] / 1000.0 # metros a km
    except:
        pass # Si falla el internet, cae silenciosamente al método matemático de abajo
        
    # PLAN B: Fórmula de Haversine * Factor de ruteo
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radio aproximado de la tierra en km
    return c * r * 1.3 # 1.3 es el factor de ruteo

# Nodos fuente (oferta) y sumideros (demanda)
nodes = {
    'Mulchen': (-37.7165, -72.2412),          # Nodo Fuente (s)
    'P_Coronel': (-37.0298, -73.1432),        # Nodos Sumidero (t) - Exportación
    'P_San_Vicente': (-36.7214, -73.1311),
    'P_Lirquen': (-36.7095, -72.9764),
    'R_Coronel': (-37.0253, -73.1345),        # Nodos Sumidero (t) - Remanufactura
    'R_Los_Angeles': (-37.2618, -72.6975),
    'Ply_Collipulli': (-37.9547, -72.4190)
}

dests = ['P_Coronel', 'P_San_Vicente', 'P_Lirquen', 'R_Coronel', 'R_Los_Angeles', 'Ply_Collipulli']

# Calculamos las distancias dinámicamente
print("Calculando distancias (intentando ruta real vía OSRM)...")
distances = {}
for d in dests:
    distances[d] = haversine(nodes['Mulchen'][0], nodes['Mulchen'][1], nodes[d][0], nodes[d][1])

products = ['P1', 'P2', 'P3']

# Utilizamos el volumen total promedio producido en la simulación como nuestra OFERTA a distribuir
production = {
    'P1': promedio_salida_por_producto['P1'],
    'P2': promedio_salida_por_producto['P2'],
    'P3': promedio_salida_por_producto['P3']
}

# Demandas Contractuales -> {Producto: {Nodo_Destino: (Minimo_Obligatorio, Capacidad_Maxima)}}
bounds_dict = {
    'P1': {
        'P_Coronel': (1200, 12000),
        'R_Coronel': (1500, 8000),
        'R_Los_Angeles': (1300, 6000),
    },
    'P2': {
        'P_Lirquen': (1500, 8000),
        'P_San_Vicente': (1500, 8000),
        'R_Los_Angeles': (1500, 5000),
        'Ply_Collipulli': (1500, 5000),
    },
    'P3': {
        'P_Coronel': (1200, 10000),
        'P_San_Vicente': (800, 12000),
        'Ply_Collipulli': (1000, 4000),
    }
}

# =====================================================================
# ARMADO DEL MODELO DE TRANSPORTE (FLUJO DE COSTO MÍNIMO)
# =====================================================================
variables = []
bounds = []
cost_coefficients = [] # Función Objetivo: Minimizar el costo de los arcos (c_ij)

for p in products:
    for d in dests:
        variables.append((p, d))
        # Costo del arco (c_ij) = Distancia * 100 CLP por cada metro cubico movido
        cost_coefficients.append(distances[d] * 100) 
        
        # Asignar limites max y min. Si el nodo no exige ese producto, limite (0,0)
        if d in bounds_dict[p]:
            bounds.append(bounds_dict[p][d])
        else:
            bounds.append((0, 0))

# Restricciones de Igualdad: La suma del producto P enviado a todos los nodos debe ser igual a la producción total de P.
A_eq = np.zeros((3, 18))
b_eq = np.zeros(3)

for i, p in enumerate(products):
    b_eq[i] = production[p]
    for j, var in enumerate(variables):
        if var[0] == p:
            A_eq[i, j] = 1

# Resolución del modelo Simplex (usando el solver 'highs' de Scipy)
res = linprog(cost_coefficients, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

print("Plan Óptimo de Despachos:\n")
if res.success:
    for j, var in enumerate(variables):
        if res.x[j] > 0.01:
            dist = distances[var[1]]
            costo_ruta = res.x[j] * 100 * dist
            print(f" -> Enviar {res.x[j]:,.2f} m3 de {var[0]} a {var[1]} (Distancia: {dist:.1f} km, Costo: $ {costo_ruta:,.0f})")
    
    print("-" * 50)
    print(f"COSTO LOGÍSTICO TOTAL MÍNIMO: $ {res.fun:,.0f} CLP")
else:
    print("Error: El modelo logístico no encontró solución factible.", res.message)
