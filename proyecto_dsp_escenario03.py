
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


# =====================================================================
# 5. PROPUESTA DE MEJORA CUANTIFICADA
# =====================================================================
print("\nCuantificando propuesta de mejora: agregar capacidad de Secado...")
print("Escenario what-if: agregar una capacidad equivalente adicional de Secado.")
print("Importante: esta es una cota estimada con tasas historicas; no es una nueva simulacion.")

improvement_summary = pd.DataFrame()
scenario_production = {}
products_for_improvement = ['P1', 'P2', 'P3']

if 'BLOCKED' not in pivot_states.columns:
    pivot_states['BLOCKED'] = 0

aserradero_state_rows = pivot_states[pivot_states['station'] == 'aserradero']
aserradero_blocked_h = aserradero_state_rows['BLOCKED'].sum()
aserradero_busy_h = aserradero_state_rows['BUSY'].sum()
aserradero_batches = batches_valid[batches_valid['station'] == 'aserradero'].copy()
aserradero_output_total = aserradero_batches['volume_out_m3'].sum()

if aserradero_busy_h > 0 and aserradero_output_total > 0:
    aserradero_rate_m3_h = aserradero_output_total / aserradero_busy_h
    extra_aserradero_output_total = aserradero_blocked_h * aserradero_rate_m3_h

    aserradero_output_by_product = (
        aserradero_batches.groupby('product')['volume_out_m3']
        .sum()
        .reindex(products_for_improvement, fill_value=0)
    )
    product_mix_aserradero = aserradero_output_by_product / aserradero_output_total

    finished_output_by_product = (
        product_outputs_valid.groupby('product')['volume_m3']
        .sum()
        .reindex(products_for_improvement, fill_value=0)
    )
    terminal_yield = (
        finished_output_by_product / aserradero_output_by_product.replace(0, np.nan)
    ).fillna(0)

    extra_finished_by_product = extra_aserradero_output_total * product_mix_aserradero * terminal_yield
    current_finished_avg = finished_output_by_product / num_replications
    extra_finished_avg = extra_finished_by_product / num_replications
    scenario_finished_avg = current_finished_avg + extra_finished_avg
    scenario_production = scenario_finished_avg.to_dict()

    improvement_summary = pd.DataFrame({
        'Producto': products_for_improvement,
        'Produccion_actual_m3': current_finished_avg.values,
        'Produccion_adicional_estimada_m3': extra_finished_avg.values,
        'Produccion_escenario_m3': scenario_finished_avg.values,
    })

    total_current = current_finished_avg.sum()
    total_extra = extra_finished_avg.sum()
    total_scenario = scenario_finished_avg.sum()
    total_increase_pct = (total_extra / total_current) * 100 if total_current > 0 else 0

    secado_output_total = batches_valid[batches_valid['station'] == 'secado']['volume_out_m3'].sum()
    extra_material_to_secado = extra_aserradero_output_total * product_mix_aserradero[['P2', 'P3']].sum()
    secado_extra_capacity_use = (
        (extra_material_to_secado / secado_output_total) * 100
        if secado_output_total > 0 else 0
    )

    print(f"Horas bloqueadas observadas en Aserradero: {aserradero_blocked_h/num_replications:,.2f} h promedio por replica")
    print(f"Tasa observada del Aserradero en BUSY: {aserradero_rate_m3_h:,.2f} m3/h")
    print(improvement_summary.to_string(
        index=False,
        formatters={
            'Produccion_actual_m3': '{:,.2f}'.format,
            'Produccion_adicional_estimada_m3': '{:,.2f}'.format,
            'Produccion_escenario_m3': '{:,.2f}'.format,
        }
    ))
    print(f"Aumento total estimado: {total_extra:,.2f} m3 por replica ({total_increase_pct:.2f}%).")
    print(f"Produccion total estimada con mejora: {total_scenario:,.2f} m3 por replica.")
    print(f"La capacidad adicional de Secado usaria aprox. {secado_extra_capacity_use:.2f}% de una capacidad equivalente actual para absorber P2/P3 extra.")
else:
    print("No hay datos suficientes para estimar la mejora de capacidad de Secado.")


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

    # =====================================================================
    # COMPARACION CON HEURISTICA DE REFERENCIA
    # =====================================================================
    print("\nComparacion con heuristica de referencia: minimos y destino factible mas cercano")

    heuristic_allocations = []
    heuristic_cost = 0.0
    heuristic_feasible = True

    for p in products:
        product_supply = production[p]
        product_bounds = bounds_dict[p]
        min_required = sum(lb for lb, _ in product_bounds.values())
        max_capacity = sum(ub for _, ub in product_bounds.values())

        if product_supply < min_required:
            print(f"ADVERTENCIA: {p} produce {product_supply:,.2f} m3, bajo el minimo requerido de {min_required:,.2f} m3.")
            heuristic_feasible = False
            continue

        if product_supply > max_capacity:
            print(f"ADVERTENCIA: {p} produce {product_supply:,.2f} m3, sobre la capacidad maxima de {max_capacity:,.2f} m3.")
            heuristic_feasible = False
            continue

        product_allocation = {d: 0.0 for d in product_bounds}

        for d, (lower_bound, _) in product_bounds.items():
            product_allocation[d] = lower_bound

        remaining_volume = product_supply - min_required
        sorted_destinations = sorted(product_bounds, key=lambda d: distances[d])

        for d in sorted_destinations:
            _, upper_bound = product_bounds[d]
            available_capacity = upper_bound - product_allocation[d]
            assigned_volume = min(remaining_volume, available_capacity)

            if assigned_volume > 0:
                product_allocation[d] += assigned_volume
                remaining_volume -= assigned_volume

            if remaining_volume <= 1e-6:
                break

        if remaining_volume > 1e-6:
            print(f"ADVERTENCIA: La heuristica no pudo asignar {remaining_volume:,.2f} m3 de {p}.")
            heuristic_feasible = False

        for d, volume in product_allocation.items():
            if volume > 0.01:
                route_cost = volume * 100 * distances[d]
                heuristic_cost += route_cost
                heuristic_allocations.append({
                    'Producto': p,
                    'Destino': d,
                    'Volumen_m3': volume,
                    'Distancia_km': distances[d],
                    'Costo_CLP': route_cost,
                })

    if heuristic_allocations:
        heuristic_plan = pd.DataFrame(heuristic_allocations)
        print("\nPlan heuristico de referencia:")
        print(heuristic_plan.to_string(
            index=False,
            formatters={
                'Volumen_m3': '{:,.2f}'.format,
                'Distancia_km': '{:.1f}'.format,
                'Costo_CLP': '$ {:,.0f}'.format,
            }
        ))

        heuristic_totals = heuristic_plan.groupby('Producto')['Volumen_m3'].sum()
        for p in products:
            shipped_volume = heuristic_totals.get(p, 0.0)
            if abs(shipped_volume - production[p]) > 1e-4:
                print(f"ADVERTENCIA: La heuristica envio {shipped_volume:,.2f} m3 de {p}, pero la produccion es {production[p]:,.2f} m3.")
                heuristic_feasible = False

        for _, row in heuristic_plan.iterrows():
            lower_bound, upper_bound = bounds_dict[row['Producto']][row['Destino']]
            if row['Volumen_m3'] < lower_bound - 1e-4 or row['Volumen_m3'] > upper_bound + 1e-4:
                print(
                    f"ADVERTENCIA: La heuristica viola limites en {row['Producto']} - {row['Destino']} "
                    f"({row['Volumen_m3']:,.2f} m3 fuera de [{lower_bound:,.2f}, {upper_bound:,.2f}])."
                )
                heuristic_feasible = False

        optimal_cost = res.fun
        cost_difference = heuristic_cost - optimal_cost
        gap_pct = (cost_difference / optimal_cost) * 100 if optimal_cost > 0 else 0

        print("-" * 50)
        print(f"COSTO HEURISTICO DE REFERENCIA: $ {heuristic_cost:,.0f} CLP")
        print(f"COSTO OPTIMO: $ {optimal_cost:,.0f} CLP")
        print(f"DIFERENCIA HEURISTICA - OPTIMO: $ {cost_difference:,.0f} CLP ({gap_pct:.2f}%)")

        if heuristic_cost + 1e-6 < optimal_cost:
            print("ADVERTENCIA: el costo heuristico quedo bajo el optimo; revisar formulacion o restricciones.")
        elif heuristic_feasible:
            print("La heuristica es factible y sirve como solucion de referencia para comparar el plan optimo.")

        # Heuristica adicional simple: no usa solver ni busqueda combinatoria.
        # Cumple minimos y reparte el excedente segun la capacidad remanente de cada destino.
        print("\nHeuristica simple adicional: reparto proporcional por capacidad remanente")
        print("Fundamento: se agrega como benchmark operativo de baja complejidad; no requiere optimizacion matematica ni alta demanda computacional.")
        print("Primero cumple minimos comerciales y luego reparte el excedente proporcionalmente a la capacidad disponible.")

        simple_heuristic_allocations = []
        simple_heuristic_cost = 0.0
        simple_heuristic_feasible = True

        for p in products:
            product_supply = production[p]
            product_bounds = bounds_dict[p]
            min_required = sum(lb for lb, _ in product_bounds.values())
            remaining_volume = product_supply - min_required

            if remaining_volume < -1e-6:
                print(f"ADVERTENCIA: {p} no alcanza los minimos para la heuristica proporcional.")
                simple_heuristic_feasible = False
                continue

            remaining_capacities = {
                d: upper_bound - lower_bound
                for d, (lower_bound, upper_bound) in product_bounds.items()
            }
            total_remaining_capacity = sum(remaining_capacities.values())

            if remaining_volume > total_remaining_capacity + 1e-6:
                print(f"ADVERTENCIA: {p} supera la capacidad disponible para la heuristica proporcional.")
                simple_heuristic_feasible = False
                continue

            for d, (lower_bound, _) in product_bounds.items():
                proportional_extra = (
                    remaining_volume * remaining_capacities[d] / total_remaining_capacity
                    if total_remaining_capacity > 0 else 0
                )
                assigned_volume = lower_bound + proportional_extra
                route_cost = assigned_volume * 100 * distances[d]
                simple_heuristic_cost += route_cost

                if assigned_volume > 0.01:
                    simple_heuristic_allocations.append({
                        'Producto': p,
                        'Destino': d,
                        'Volumen_m3': assigned_volume,
                        'Distancia_km': distances[d],
                        'Costo_CLP': route_cost,
                    })

        if simple_heuristic_allocations:
            simple_heuristic_plan = pd.DataFrame(simple_heuristic_allocations)
            print("\nPlan heuristico proporcional:")
            print(simple_heuristic_plan.to_string(
                index=False,
                formatters={
                    'Volumen_m3': '{:,.2f}'.format,
                    'Distancia_km': '{:.1f}'.format,
                    'Costo_CLP': '$ {:,.0f}'.format,
                }
            ))

            simple_totals = simple_heuristic_plan.groupby('Producto')['Volumen_m3'].sum()
            for p in products:
                shipped_volume = simple_totals.get(p, 0.0)
                if abs(shipped_volume - production[p]) > 1e-4:
                    print(f"ADVERTENCIA: La heuristica proporcional envio {shipped_volume:,.2f} m3 de {p}, pero la produccion es {production[p]:,.2f} m3.")
                    simple_heuristic_feasible = False

            for _, row in simple_heuristic_plan.iterrows():
                lower_bound, upper_bound = bounds_dict[row['Producto']][row['Destino']]
                if row['Volumen_m3'] < lower_bound - 1e-4 or row['Volumen_m3'] > upper_bound + 1e-4:
                    print(
                        f"ADVERTENCIA: La heuristica proporcional viola limites en {row['Producto']} - {row['Destino']} "
                        f"({row['Volumen_m3']:,.2f} m3 fuera de [{lower_bound:,.2f}, {upper_bound:,.2f}])."
                    )
                    simple_heuristic_feasible = False

            simple_cost_difference = simple_heuristic_cost - optimal_cost
            simple_gap_pct = (simple_cost_difference / optimal_cost) * 100 if optimal_cost > 0 else 0

            print("-" * 50)
            print(f"COSTO HEURISTICO PROPORCIONAL: $ {simple_heuristic_cost:,.0f} CLP")
            print(f"COSTO OPTIMO: $ {optimal_cost:,.0f} CLP")
            print(f"DIFERENCIA PROPORCIONAL - OPTIMO: $ {simple_cost_difference:,.0f} CLP ({simple_gap_pct:.2f}%)")

            if simple_heuristic_cost + 1e-6 < optimal_cost:
                print("ADVERTENCIA: el costo proporcional quedo bajo el optimo; revisar formulacion o restricciones.")
            elif simple_heuristic_feasible:
                print("La heuristica proporcional es factible y muestra el costo de una politica simple que no prioriza distancia.")
    else:
        print("No se pudo construir un plan heuristico de referencia.")
else:
    print("Error: El modelo logístico no encontró solución factible.", res.message)
