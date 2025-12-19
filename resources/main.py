import pandas as pd
from tkinter import Tk, filedialog
import tkinter as tk
import random
import numpy as np
from tkinter import ttk
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from itertools import combinations

datos = None


def cerrar_todo():
    root.destroy()  # cierra TODAS las ventanas y termina el programa


def volver(ventana_actual, ventana_anterior):
    ventana_actual.destroy()
    ventana_anterior.deiconify()


##Calculo MAS
def calcula_mas():
    calculo_mas = tk.Toplevel(mas)
    calculo_mas.title("Cálculo por MAS")
    calculo_mas.geometry("1000x900")
    datos_mas = datos
    col_var = entrada_var.get()
    tamanio_muestra = int(entrada_tamanio.get())
    longitud_columna = len(datos_mas[col_var])
    mas.withdraw()
    unidades_muestra = random.sample(range(1, longitud_columna + 1), tamanio_muestra)
    valores = np.array([datos_mas.iloc[i - 1][col_var] for i in unidades_muestra])
    suma = np.sum(valores)
    cuadrados = valores**2
    suma_cuadrados = np.sum(cuadrados)
    estimador_total = longitud_columna * (suma / tamanio_muestra)
    estimador_promedio = suma / tamanio_muestra
    s_cua_chic = (1 / (tamanio_muestra - 1)) * (
        suma_cuadrados - ((1 / tamanio_muestra) * (suma**2))
    )
    est_var_total = (
        (longitud_columna**2)
        * (1 - (tamanio_muestra / longitud_columna))
        * (s_cua_chic / tamanio_muestra)
    )
    est_var_prom = (1 - (tamanio_muestra / longitud_columna)) * (
        s_cua_chic / tamanio_muestra
    )
    err_est_total = math.sqrt(est_var_total)
    err_est_prom = math.sqrt(est_var_prom)

    ##---------Muestra-------------#
    df_muestra = datos_mas[datos_mas["Id"].isin(unidades_muestra)]
    marco_df_muestra = ttk.LabelFrame(calculo_mas, text="Datos Muestra", padding=10)
    marco_df_muestra.pack(fill="both", expand=True, padx=10, pady=10)
    tabla_muestra = ttk.Treeview(
        marco_df_muestra, columns=list(df_muestra.columns), show="headings", height=5
    )
    tabla_muestra.pack(fill="both", expand=True)
    for col in df_muestra.columns:
        tabla_muestra.heading(col, text=col)
        tabla_muestra.column(col, width=120)
    for _, fila in df_muestra.iterrows():
        tabla_muestra.insert("", "end", values=list(fila))
    scroll_y = ttk.Scrollbar(
        marco_df_muestra, orient="vertical", command=tabla_muestra.yview
    )
    tabla_muestra.configure(yscrollcommand=scroll_y.set)
    scroll_y.pack(side="right", fill="y")
    ##-----------------
    ##-------Resultados
    marco_res = ttk.LabelFrame(calculo_mas, text="Resultados", padding=10)
    marco_res.pack(fill="x", padx=10, pady=10)
    ttk.Label(marco_res, text=f"Estimador Total: {estimador_total}").pack(
        anchor="w", pady=3
    )
    ttk.Label(marco_res, text=f"Estimador Promedio: {estimador_promedio}").pack(
        anchor="w", pady=3
    )
    ttk.Label(
        marco_res,
        text=f"Estimador de Varianza del estimador del total: {est_var_total}",
    ).pack(anchor="w", pady=3)
    ttk.Label(
        marco_res,
        text=f"Estimador de Varianza del estimador del promedio: {est_var_prom}",
    ).pack(anchor="w", pady=3)
    ttk.Label(marco_res, text=f"Estimador Error est Total: {err_est_total}").pack(
        anchor="w", pady=3
    )
    ttk.Label(marco_res, text=f"Estimador Error est promedio: {err_est_prom}").pack(
        anchor="w", pady=3
    )
    ##------------------
    boton_volver = tk.Button(
        calculo_mas,
        text="⬅ Regresar a especifícaciones MAS",
        command=lambda: volver(calculo_mas, mas),
    )
    boton_volver.place(x=600, y=580)
    calculo_mas.protocol("WM_DELETE_WINDOW", cerrar_todo)


##Calculo ER
def calcula_er():
    calculo_er = tk.Toplevel(er)
    calculo_er.title("Cálculo por Estimador Razón")
    calculo_er.geometry("1000x900")
    datos_er = datos
    col_var = entrada_var.get()
    col_var_aux = entrada_var_aux.get()
    tamanio_muestra = int(entrada_tamanio.get())
    longitud_columna = len(datos_er[col_var])
    er.withdraw()
    unidades_muestra = random.sample(range(1, longitud_columna + 1), tamanio_muestra)
    valores_prim = np.array([datos_er.iloc[i - 1][col_var] for i in unidades_muestra])
    valores_aux = np.array(
        [datos_er.iloc[j - 1][col_var_aux] for j in unidades_muestra]
    )
    valor_cruz = valores_prim * valores_aux
    sum_valor_cruz = np.sum(valor_cruz)
    sum_primarios = np.sum(valores_prim)
    sum_aux = np.sum(valores_aux)
    cuadrados = valores_prim**2
    cuadrados_aux = valores_aux**2
    sum_cua_prim = np.sum(cuadrados)
    sum_cua_aux = np.sum(cuadrados_aux)
    aux_total = datos_er[col_var_aux].sum()
    aux_prom = aux_total / longitud_columna
    r_gorr = sum_primarios / sum_aux
    est_total = (aux_total) * (sum_primarios / sum_aux)
    est_promedio = (aux_prom) * (sum_primarios / sum_aux)
    sr_chic = (1 / (tamanio_muestra - 1)) * (
        sum_cua_prim - (2 * r_gorr * (sum_valor_cruz)) + ((r_gorr**2) * sum_cua_aux)
    )
    var_total = (
        (longitud_columna**2)
        * (1 - (tamanio_muestra / longitud_columna))
        * (sr_chic / tamanio_muestra)
    )
    var_prom = (1 - (tamanio_muestra / longitud_columna)) * (sr_chic / tamanio_muestra)
    err_est_total = math.sqrt(var_total)
    err_est_prom = math.sqrt(var_prom)
    ##---------Muestra-------------#
    df_muestra = datos_er[datos_er["Id"].isin(unidades_muestra)]
    marco_df_muestra = ttk.LabelFrame(calculo_er, text="Datos Muestra", padding=10)
    marco_df_muestra.pack(fill="both", expand=True, padx=10, pady=10)
    tabla_muestra = ttk.Treeview(
        marco_df_muestra, columns=list(df_muestra.columns), show="headings", height=5
    )
    tabla_muestra.pack(fill="both", expand=True)
    for col in df_muestra.columns:
        tabla_muestra.heading(col, text=col)
        tabla_muestra.column(col, width=120)
    for _, fila in df_muestra.iterrows():
        tabla_muestra.insert("", "end", values=list(fila))
    scroll_y = ttk.Scrollbar(
        marco_df_muestra, orient="vertical", command=tabla_muestra.yview
    )
    tabla_muestra.configure(yscrollcommand=scroll_y.set)
    scroll_y.pack(side="right", fill="y")
    ##-----------------
    ##-------Resultados
    marco_res = ttk.LabelFrame(calculo_er, text="Resultados", padding=10)
    marco_res.pack(fill="x", padx=10, pady=10)
    ttk.Label(marco_res, text=f"Estimador Total: {est_total}").pack(anchor="w", pady=3)
    ttk.Label(marco_res, text=f"Estimador Promedio: {est_promedio}").pack(
        anchor="w", pady=3
    )
    ttk.Label(
        marco_res,
        text=f"Estimador de Varianza del estimador del total: {var_total}",
    ).pack(anchor="w", pady=3)
    ttk.Label(
        marco_res,
        text=f"Estimador de Varianza del estimador del promedio: {var_prom}",
    ).pack(anchor="w", pady=3)
    ttk.Label(marco_res, text=f"Estimador Error est Total: {err_est_total}").pack(
        anchor="w", pady=3
    )
    ttk.Label(marco_res, text=f"Estimador Error est promedio: {err_est_prom}").pack(
        anchor="w", pady=3
    )
    ##------------------
    boton_volver = tk.Button(
        calculo_er,
        text="⬅ Regresar a especifícaciones ER",
        command=lambda: volver(calculo_er, er),
    )
    boton_volver.place(x=600, y=580)
    calculo_er.protocol("WM_DELETE_WINDOW", cerrar_todo)


def calcula_est():
    calculo_est = tk.Toplevel()
    calculo_est.title("Cálculo por Estimador Razón")
    calculo_est.geometry("1000x900")
    datos_est = datos
    col_var = entrada_var.get()
    longitud_columna = len(datos_est[col_var])
    tamanio_muestra = int(entrada_tamanio.get())
    col_estratos = entrada_est.get()
    estratos = estratos = [
        x.strip() for x in entrada_estratos_cu.get().strip("()").split(",")
    ]
    est.withdraw()
    num_cada_est = []
    df_estratos = []
    for i in estratos:
        numero = (datos_est[col_estratos].astype(str) == i).sum()
        num_cada_est.append(numero)
        df_estratos.append(datos_est[datos_est[col_estratos].astype(str) == i])
    n_prop_est = []
    for j in num_cada_est:
        n_prop = tamanio_muestra * (j / longitud_columna)
        n_prop_est.append(n_prop)
    muestras_estratos = []
    ids_por_estrato = []

    for idx, df_est_i in enumerate(df_estratos):
        n = int(n_prop_est[idx])
        if n > 0:
            muestra_i = df_est_i.sample(n=n, random_state=1)
        else:
            muestra_i = df_est_i.head(0)

        muestras_estratos.append(muestra_i)

        ids_estrato_i = muestra_i["Id"].tolist()
        ids_por_estrato.append(ids_estrato_i)
    valores_estrato = []
    for i in ids_por_estrato:
        valores_estrato.append(np.array([datos_est.iloc[x - 1][col_var] for x in i]))
    suma = []
    for i in valores_estrato:
        suma.append(np.sum(i))
    cuadrados = []
    for i in valores_estrato:
        cuadrados.append(i**2)
    suma_cuadrados = []
    for i in cuadrados:
        suma_cuadrados.append(np.sum(i))
    est_total = []
    for i, j, z in zip(suma, num_cada_est, n_prop_est):
        est_total.append((i / z) * j)
    est_promedio = []
    for i, j, z in zip(suma, n_prop_est, num_cada_est):
        est_promedio.append((i / j) * (z / longitud_columna))
    est_total_bueno = np.sum(est_total)
    est_promedio_bueno = np.sum(est_promedio)
    var_tot = []
    for Nh, nh, valores in zip(num_cada_est, n_prop_est, valores_estrato):

        nh = int(nh)

        if nh > 1:
            Sh2 = np.var(valores, ddof=1)
            f = nh / Nh
            var_h = (Nh**2) * (1 - f) * (Sh2 / nh)
        else:
            var_h = 0

        var_tot.append(var_h)

    var_total_estimador = np.sum(var_tot)

    var_prom = []
    for Nh, nh, valores in zip(num_cada_est, n_prop_est, valores_estrato):

        nh = int(nh)

        if nh > 1:
            Sh2 = np.var(valores, ddof=1)
            f = nh / Nh
            Wh = Nh / longitud_columna
            var_h = (Wh**2) * (1 - f) * (Sh2 / nh)
        else:
            var_h = 0

        var_prom.append(var_h)
    var_promedio_estimador = np.sum(var_prom)
    err_est_prom = math.sqrt(var_promedio_estimador)
    err_est_total = math.sqrt(var_total_estimador)

    ##-------Resultados
    marco_res = ttk.LabelFrame(calculo_est, text="Resultados", padding=10)
    marco_res.pack(fill="x", padx=10, pady=10)
    ttk.Label(marco_res, text=f"Estimador Total: {est_total_bueno}").pack(
        anchor="w", pady=3
    )
    ttk.Label(marco_res, text=f"Estimador Promedio: {est_promedio_bueno}").pack(
        anchor="w", pady=3
    )
    ttk.Label(
        marco_res,
        text=f"Estimador de Varianza del estimador del total: {var_total_estimador}",
    ).pack(anchor="w", pady=3)
    ttk.Label(
        marco_res,
        text=f"Estimador de Varianza del estimador del promedio: {var_promedio_estimador}",
    ).pack(anchor="w", pady=3)
    ttk.Label(marco_res, text=f"Estimador Error est Total: {err_est_total}").pack(
        anchor="w", pady=3
    )
    ttk.Label(marco_res, text=f"Estimador Error est promedio: {err_est_prom}").pack(
        anchor="w", pady=3
    )
    ##------------------
    boton_volver = tk.Button(
        calculo_est,
        text="⬅ Regresar a especifícaciones ESTRAT",
        command=lambda: volver(calculo_est, est),
    )
    boton_volver.place(x=600, y=580)
    calculo_est.protocol("WM_DELETE_WINDOW", cerrar_todo)


def calcula_comparativa():
    comparativa_ven = tk.Toplevel(comparativ)
    comparativa_ven.title("Cálculo por MAS")
    comparativa_ven.geometry("1000x900")
    datos_comp = datos
    col_var = entrada_varc.get()
    col_aux = entrada_varauxc.get()
    longitud_columna = len(datos_comp[col_var])
    num_unidades = 10
    varianzas_mas = []
    varianzas_er = []
    unidades = []
    comparativ.withdraw()
    while num_unidades <= longitud_columna:
        unidades_muestra = random.sample(range(1, longitud_columna + 1), num_unidades)
        valores = np.array([datos_comp.iloc[i - 1][col_var] for i in unidades_muestra])
        valores_aux = np.array(
            [datos_comp.iloc[i - 1][col_aux] for i in unidades_muestra]
        )
        cuadrados_val = valores**2
        cuadrados_aux = valores_aux**2
        suma_cuad_val = np.sum(cuadrados_val)
        suma_cuad_aux = np.sum(cuadrados_aux)
        suma_val = np.sum(valores)
        suma_aux = np.sum(valores_aux)
        valor_cruz_er = valores * valores_aux
        sum_valor_cruz_er = np.sum(valor_cruz_er)
        r_chica = suma_val / suma_aux
        s_cua_chic_er = (1 / (num_unidades - 1)) * (
            suma_cuad_val
            - (2 * r_chica * (sum_valor_cruz_er))
            + ((r_chica**2) * suma_cuad_aux)
        )
        s_cua_chic_mas = (1 / (num_unidades - 1)) * (
            suma_cuad_val - ((1 / num_unidades) * (suma_val**2))
        )
        est_var_prom_mas = (1 - (num_unidades / longitud_columna)) * (
            s_cua_chic_mas / num_unidades
        )
        est_var_prom_er = (1 - (num_unidades / longitud_columna)) * (
            s_cua_chic_er / num_unidades
        )
        unidades.append(num_unidades)
        varianzas_mas.append(est_var_prom_mas)
        varianzas_er.append(est_var_prom_er)
        num_unidades = num_unidades + 4
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(unidades, varianzas_mas, marker="o", label="Varianza MAS")
    ax.plot(unidades, varianzas_er, marker="s", label="Varianza ER")
    ax.set_xlabel("Número de unidades")
    ax.set_ylabel("Varianza estimada")
    ax.set_title("Comparación de Varianzas MAS vs ER")
    ax.legend()
    ax.grid(True)
    canvas = FigureCanvasTkAgg(fig, master=comparativa_ven)
    canvas.draw()
    canvas.get_tk_widget().place(x=50, y=50)
    boton_volver = tk.Button(
        comparativa_ven,
        text="⬅ Regresar a especifícaciones Comparativa",
        command=lambda: volver(comparativa_ven, comparativ),
    )
    boton_volver.place(x=100, y=700)
    comparativa_ven.protocol("WM_DELETE_WINDOW", cerrar_todo)


def calc_opt():
    optimo_mue = tk.Toplevel(opt)
    optimo_mue.title("Calculadora de número de unidades óptimo para la muestra")
    optimo_mue.geometry("1000x900")

    z = float(entrada_z.get())
    s_cuadrada = float(entrada_var_est.get())
    error_absoluto = float(entrada_err_abs.get())
    tipo = (entrada_tipo.get()).upper
    num_pob = float(entrada_num_datos.get())
    opt.withdraw()
    if tipo == "PROMEDIO":
        num_opt = 1 / (((error_absoluto**2) / ((z**2) * s_cuadrada)) + (1 / num_pob))
    else:
        num_opt = 1 / (
            ((error_absoluto**2) / ((z**2) * s_cuadrada * (num_pob**2))) + (1 / num_pob)
        )

    numero_optimo = tk.Label(
        optimo_mue, text=f"El número de unidades en muestra óptimo es: {num_opt}"
    )
    numero_optimo.place(x=20, y=40)

    boton_volver = tk.Button(
        optimo_mue,
        text="⬅ Regresar a Selección método",
        command=lambda: volver(optimo_mue, opt),
    )
    boton_volver.place(x=20, y=100)
    optimo_mue.protocol("WM_DELETE_WINDOW", cerrar_todo)


def dist_calc():
    dist_calc = tk.Toplevel(distribucion)
    dist_calc.geometry("1000x900")
    dist_calc.title("Distribución")
    datos_dist = datos
    tamanio_muestra = int(taman.get())
    columna = var.get()
    distribucion.withdraw()
    rango = range(1, len(datos[columna]) + 1)
    proba_de_est = 1 / (math.comb(len(datos[columna]) + 1, tamanio_muestra))
    lista_est = []
    esperanza = 0
    for i in combinations(rango, tamanio_muestra):
        lista_unidades = list(i)
        valores = np.array([datos_dist.iloc[j - 1][columna] for j in lista_unidades])
        suma_valores = np.sum(valores)
        estimador_prom = suma_valores / tamanio_muestra
        esperanza = esperanza + (estimador_prom * proba_de_est)
        lista_est.append(estimador_prom)
    est_num = []
    for est in lista_est:
        num_veces = lista_est.count(est)
        est_num.append([est, num_veces])
    lista_est_val = list(dict.fromkeys(est_num))
    x_vals = [par[0] for par in lista_est_val]
    y_vals = [par[1] for par in lista_est_val]
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(x_vals, y_vals)
    ax.set_xlabel("Estimador")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución del estimador")
    canvas = FigureCanvasTkAgg(fig, master=dist_calc)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    boton_volver = tk.Button(
        dist_calc,
        text="⬅ Regresar a Selección método",
        command=lambda: volver(dist_calc, distribucion),
    )
    boton_volver.place(x=20, y=2000)


##Ventana MAS
def est_MAS():
    global entrada_var, entrada_tamanio, mas
    mas = tk.Toplevel(segunda)
    segunda.withdraw()
    mas.title("Estimación por MAS")
    mas.geometry("1000x900")
    label_titulo = tk.Label(mas, text="Estimación por Muestreo Aleatorio Simple")
    label_titulo.place(x=20, y=20)
    label_variable_est = tk.Label(mas, text="Nombre de Columna de Variable a estimar")
    label_variable_est.place(x=20, y=40)
    entrada_var = tk.Entry(mas)
    entrada_var.place(x=320, y=40)
    label_tamanio_muestra = tk.Label(mas, text="Tamaño muestra")
    label_tamanio_muestra.place(x=20, y=60)
    entrada_tamanio = tk.Entry(mas)
    entrada_tamanio.place(x=320, y=60)
    boton_calcular = tk.Button(mas, text="Calcular", command=calcula_mas)
    boton_calcular.place(x=20, y=80)
    boton_volver = tk.Button(
        mas, text="⬅ Regresar a Selección método", command=lambda: volver(mas, segunda)
    )
    boton_volver.place(x=20, y=120)
    mas.protocol("WM_DELETE_WINDOW", cerrar_todo)


##Ventana ER
def est_ER():
    global entrada_var, entrada_tamanio, entrada_var_aux, er
    er = tk.Toplevel(segunda)
    segunda.withdraw()
    er.title("Estimación por Estimador de Razón")
    er.geometry("1000x900")
    label_titulo = tk.Label(er, text="Estimación por Estimador de Razón")
    label_titulo.place(x=20, y=20)
    label_variable_est = tk.Label(er, text="Nombre de Columna de Variable a estimar")
    label_variable_est.place(x=20, y=40)
    entrada_var = tk.Entry(er)
    entrada_var.place(x=320, y=40)
    label_variable_aux = tk.Label(er, text="Nombre de Columna Variable auxiliar")
    label_variable_aux.place(x=20, y=60)
    entrada_var_aux = tk.Entry(er)
    entrada_var_aux.place(x=320, y=60)
    label_tamanio_muestra = tk.Label(er, text="Tamaño muestra")
    label_tamanio_muestra.place(x=20, y=80)
    entrada_tamanio = tk.Entry(er)
    entrada_tamanio.place(x=320, y=80)
    boton_calcular = tk.Button(er, text="Calcular", command=calcula_er)
    boton_calcular.place(x=20, y=100)
    boton_volver = tk.Button(
        er, text="⬅ Regresar a Selección método", command=lambda: volver(er, segunda)
    )
    boton_volver.place(x=20, y=140)
    er.protocol("WM_DELETE_WINDOW", cerrar_todo)


##Ventana EST
def est_EST():
    global entrada_var, entrada_tamanio, entrada_est, est, entrada_estratos_cu
    est = tk.Toplevel(segunda)
    segunda.withdraw()
    est.title("Estimación por Estratificación")
    est.geometry("1000x900")
    label_titulo = tk.Label(est, text="Estimación por Estratificación")
    label_titulo.place(x=20, y=20)
    label_variable_est = tk.Label(est, text="Nombre de Columna de Variable a estimar")
    label_variable_est.place(x=20, y=40)
    entrada_var = tk.Entry(est)
    entrada_var.place(x=320, y=40)
    label_variable_est = tk.Label(est, text="Nombre de Columna con Estratos")
    label_variable_est.place(x=20, y=60)
    entrada_est = tk.Entry(est)
    entrada_est.place(x=320, y=60)
    label_tamanio_muestra = tk.Label(est, text="Tamaño muestra")
    label_tamanio_muestra.place(x=20, y=80)
    entrada_tamanio = tk.Entry(est)
    entrada_tamanio.place(x=320, y=80)
    label_estratos_cu = tk.Label(
        est,
        text="¿Qué estratos hay? EJ. Si estratos estan en tabla como H hombre y M mujer (H,M)",
    )
    label_estratos_cu.place(x=20, y=100)
    entrada_estratos_cu = tk.Entry(est)
    entrada_estratos_cu.place(x=560, y=100)
    boton_calcular = tk.Button(est, text="Calcular", command=calcula_est)
    boton_calcular.place(x=20, y=140)
    boton_volver = tk.Button(
        est, text="⬅ Regresar a Selección método", command=lambda: volver(est, segunda)
    )
    boton_volver.place(x=20, y=180)
    est.protocol("WM_DELETE_WINDOW", cerrar_todo)


##Ventana Comparativa
def comparativa():
    global comparativ, entrada_varc, entrada_varauxc
    comparativ = tk.Toplevel(segunda)
    segunda.withdraw()
    comparativ.title("Comparativa Varianzas MAS vs ER")
    comparativ.geometry("1000x900")
    label_titulo = tk.Label(comparativ, text="Comparativa Varianzas MAS vs ER")
    label_titulo.place(x=20, y=20)
    label_variable_est = tk.Label(
        comparativ, text="Nombre de Columna de Variable a estimar"
    )
    label_variable_est.place(x=20, y=40)
    entrada_varc = tk.Entry(comparativ)
    entrada_varc.place(x=320, y=40)
    label_variable_aux = tk.Label(comparativ, text="Nombre Columna auxiliar (en ER)")
    label_variable_aux.place(x=20, y=60)
    entrada_varauxc = tk.Entry(comparativ)
    entrada_varauxc.place(x=320, y=60)
    boton_calcular = tk.Button(comparativ, text="Calcular", command=calcula_comparativa)
    boton_calcular.place(x=20, y=80)
    boton_volver = tk.Button(
        comparativ,
        text="⬅ Regresar a Selección método",
        command=lambda: volver(comparativ, segunda),
    )
    boton_volver.place(x=20, y=120)
    comparativ.protocol("WM_DELETE_WINDOW", cerrar_todo)


##Ventana Calculadora Tamaño Muestra
def optimo_muestra():
    global opt, entrada_z, entrada_err_abs, entrada_tipo, entrada_num_datos, entrada_var_est
    opt = tk.Toplevel(segunda)
    segunda.withdraw()
    opt.title("Calculadora de número de unidades en Muestra")
    opt.geometry("1000x900")

    frame = tk.Frame(opt)
    frame.pack(anchor="center", pady=20)
    style = ttk.Style()
    style.configure("Compact.Treeview", font=("Arial", 9), rowheight=18)

    columnas = ("nivel", "alpha", "z")
    tabla = ttk.Treeview(
        frame, columns=columnas, show="headings", height=5, style="Compact.Treeview"
    )

    tabla.heading("nivel", text="1 - α")
    tabla.heading("alpha", text="α")
    tabla.heading("z", text="Z")

    tabla.column("nivel", anchor="center", width=90)
    tabla.column("alpha", anchor="center", width=60)
    tabla.column("z", anchor="center", width=60)

    datos = [
        (0.90, 0.10, 1.282),
        (0.95, 0.05, 1.645),
        (0.975, 0.025, 1.960),
        (0.99, 0.01, 2.326),
        (0.995, 0.005, 2.576),
    ]

    for fila in datos:
        tabla.insert("", tk.END, values=fila)
    tabla.pack(anchor="w")
    label_err_abs = tk.Label(opt, text="Error absoluto")
    label_err_abs.place(x=20, y=180)
    entrada_err_abs = tk.Entry(opt)
    entrada_err_abs.place(x=300, y=180)
    label_varianza_est = tk.Label(opt, text="Varianza estimada de datos")
    label_varianza_est.place(x=20, y=200)
    entrada_var_est = tk.Entry(opt)
    entrada_var_est.place(x=300, y=200)
    label_num_datos = tk.Label(opt, text="Número unidades en población")
    label_num_datos.place(x=20, y=220)
    entrada_num_datos = tk.Entry(opt)
    entrada_num_datos.place(x=300, y=220)
    label_nivel_z = tk.Label(opt, text="Nivel confianza Z")
    label_nivel_z.place(x=20, y=240)
    entrada_z = tk.Entry(opt)
    entrada_z.place(x=300, y=240)
    label_tipo = tk.Label(opt, text="¿Para que estimación Total o Promedio?")
    label_tipo.place(x=20, y=260)
    entrada_tipo = tk.Entry(opt)
    entrada_tipo.place(x=300, y=260)
    boton_calc = tk.Button(opt, text="Calcular", command=calc_opt)
    boton_calc.place(x=20, y=300)
    boton_volver = tk.Button(
        opt,
        text="⬅ Regresar a Selección método",
        command=lambda: volver(opt, segunda),
    )
    boton_volver.place(x=20, y=340)
    opt.protocol("WM_DELETE_WINDOW", cerrar_todo)


def distribucion():
    global distribucion, var, taman
    distribucion = tk.Toplevel(segunda)
    segunda.withdraw()
    distribucion.title("Distribución de muestra para estimador ")
    distribucion.geometry("1000x900")
    label_titulo = tk.Label(
        distribucion, text="Distribución de muestra para estimador de promedio"
    )
    label_titulo.place(x=20, y=20)
    label_variable_est = tk.Label(
        distribucion, text="Nombre de Columna de Variable a estimar"
    )
    label_variable_est.place(x=20, y=40)
    var = tk.Entry(distribucion)
    var.place(x=300, y=40)
    label_tamanio_muestra = tk.Label(distribucion, text="Tamaño muestra")
    label_tamanio_muestra.place(x=20, y=60)
    taman = tk.Entry(distribucion)
    taman.place(x=300, y=60)
    boton_calc = tk.Button(distribucion, text="Calcular", command=dist_calc)
    boton_calc.place(x=20, y=150)
    boton_volver = tk.Button(
        distribucion,
        text="⬅ Regresar a Selección método",
        command=lambda: volver(distribucion, segunda),
    )
    boton_volver.place(x=20, y=2000)
    distribucion.protocol("WM_DELETE_WINDOW", cerrar_todo)


##Ventana Secundaria
def abrir_segunda_ven():
    global segunda
    segunda = tk.Toplevel(root)
    segunda.title("Tipo de Estimación a usar")
    segunda.geometry("1000x900")
    label_archivo = tk.Label(segunda, text=f"Archivo cargado: {archivo.split('/')[-1]}")
    label_archivo.place(x=20, y=20)
    label_estrteg = tk.Label(
        segunda,
        text="Selecciona a través de que método quieres realizar la estimación estadística",
    )
    label_estrteg.place(x=20, y=40)
    boton_MAS = tk.Button(segunda, text="Estimación por MAS", command=est_MAS)
    boton_MAS.place(x=20, y=60)
    boton_ER = tk.Button(segunda, text="Estimación por Estimador Razón", command=est_ER)
    boton_ER.place(x=180, y=60)
    boton_EST = tk.Button(
        segunda, text="Estimación por Estratificación", command=est_EST
    )
    boton_EST.place(x=420, y=60)
    boton_comparativa = tk.Button(
        segunda, text="Comparar Varianzas de MAS vs ER", command=comparativa
    )
    boton_comparativa.place(x=660, y=60)
    boton_numopt = tk.Button(
        segunda, text="Número de unidades en muestra óptimo", command=optimo_muestra
    )
    boton_numopt.place(x=20, y=120)
    label_calc_num = tk.Label(
        segunda, text="Calculadora de número de unidades óptima para muestra"
    )
    label_calc_num.place(x=20, y=90)
    boton_distribucion = tk.Button(
        segunda,
        text="Distribución de estimación por tamaño de muestra",
        command=distribucion,
    )
    boton_distribucion.place(x=350, y=120)
    boton_volver = tk.Button(
        segunda, text="⬅ Regresar a inicio", command=lambda: volver(segunda, root)
    )
    boton_volver.place(x=20, y=180)
    segunda.protocol("WM_DELETE_WINDOW", cerrar_todo)


def selecciona_base():
    global datos, archivo
    archivo = filedialog.askopenfilename(
        title="Selecciona un archivo CSV o XLSX",
        filetypes=[("Archivos CSV", "*.csv"), ("Archivos Excel", "*.xlsx")],
    )
    if not archivo:
        return
    if archivo.endswith(".csv"):
        datos = pd.read_csv(archivo)
    else:
        datos = pd.read_excel(archivo)
    if "Id" not in datos.columns:
        datos.insert(0, "Id", range(1, len(datos) + 1))
    root.withdraw()
    abrir_segunda_ven()


##Ventana Principal
root = tk.Tk()
root.title("Calculadora estadística")
root.geometry("1000x900")
etiqueta_sele = tk.Label(root, text="Selecciona Base de Datos a analizar")
etiqueta_sele.place(x=20, y=20)
boton_selec = tk.Button(root, text="Seleccionar", command=selecciona_base)
boton_selec.place(x=100, y=100)
root.mainloop()
