"""Laboratorio Virtual de Sistemas Eléctricos de Potencia (SEP).

Aplicación educativa interactiva (Streamlit + Plotly) con 8 módulos sobre
control, tensión y frecuencia en sistemas de potencia.

Crear el entorno virtual
	python -m venv .venv

Activarlo
	.venv\Scripts\activate


Instalación:
    python -m pip install streamlit plotly numpy

Ejecución:
    python -m streamlit run programacion.py

Arquitectura (SRP):
    1. Utilidades y validación        -> ParameterError, require_range
    2. Capa de física (sin Streamlit) -> modelos matemáticos puros y testeables
    3. Capa de UI                     -> LabModule (clase base) + módulos registrados
    4. main()                         -> ensamblado del dashboard

Cómo ampliar el laboratorio: ver los comentarios "# EXTENSIÓN:".
Los modelos son DIDÁCTICOS (simplificados); no sustituyen estudios de red reales.
"""
from __future__ import annotations

import cmath
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Sequence, Tuple, Type

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# =============================================================================
# 1. UTILIDADES Y VALIDACIÓN
# =============================================================================


class ParameterError(ValueError):
    """Parámetro físico fuera de su rango válido."""


def require_range(name: str, value: float, lo: float, hi: float) -> float:
    """Valida que ``lo <= value <= hi``.

    Args:
        name: Nombre legible del parámetro.
        value: Valor a validar.
        lo: Límite inferior.
        hi: Límite superior.

    Returns:
        El mismo valor si es válido.

    Raises:
        ParameterError: Si el valor está fuera de rango o no es finito.
    """
    if not (math.isfinite(value) and lo <= value <= hi):
        raise ParameterError(f"{name} = {value} fuera de rango [{lo}, {hi}]")
    return value


def magnetizing_current_pu(vf_pu: float) -> float:
    """Corriente de magnetización didáctica de un transformador.

    Modelo exponencial: 1 % a V/f = 1 pu y crecimiento rápido al saturar.

    Args:
        vf_pu: Relación V/f en por unidad.

    Returns:
        Corriente de excitación en pu de la corriente nominal.
    """
    return 0.01 * math.exp(22.0 * (vf_pu - 1.0))


# =============================================================================
# 2. CAPA DE FÍSICA (independiente de la interfaz)
# =============================================================================


@dataclass(frozen=True)
class PowerBalanceModel:
    """Equilibrio estático P-f y Q-V del sistema (Módulo 1).

    Attributes:
        f0: Frecuencia nominal [Hz].
        s_base_mw: Potencia base del sistema [MW].
        droop: Estatismo equivalente R [pu].
        damping: Autorregulación de carga D [pu/pu].
        s_sc_mva: Potencia de cortocircuito en el nodo [MVA].
    """

    f0: float = 60.0
    s_base_mw: float = 1000.0
    droop: float = 0.05
    damping: float = 1.0
    s_sc_mva: float = 3000.0

    def __post_init__(self) -> None:
        require_range("Estatismo R", self.droop, 0.01, 0.12)
        require_range("Amortiguamiento D", self.damping, 0.0, 5.0)
        require_range("Potencia de cortocircuito", self.s_sc_mva, 100.0, 1e5)

    def steady_frequency(self, p_gen_mw: float, p_load_mw: float) -> float:
        """Frecuencia de régimen permanente tras la regulación primaria [Hz]."""
        beta = self.damping + 1.0 / self.droop
        return self.f0 * (1.0 + (p_gen_mw - p_load_mw) / self.s_base_mw / beta)

    def steady_voltage(self, q_gen_mvar: float, q_load_mvar: float) -> float:
        """Tensión nodal aproximada [pu]: ΔV ≈ ΔQ / S_cc."""
        return 1.0 + (q_gen_mvar - q_load_mvar) / self.s_sc_mva


class MotorStartingModel:
    """Arranque de un motor de inducción desde una red con impedancia Thevenin.

    El motor se modela como admitancia constante dependiente del deslizamiento
    s(t) = exp(-3 t / t_acc). La red es una fuente ideal con Z_cc (X/R dado).
    """

    METHODS: ClassVar[Tuple[str, ...]] = (
        "Directo", "Estrella-Triángulo", "Arrancador suave", "VFD")
    TORQUE_FACTOR: ClassVar[Dict[str, float]] = {
        "Directo": 1.0, "Estrella-Triángulo": 1 / 3, "Arrancador suave": 0.25,
        "VFD": 1.5}
    LOCKED_TORQUE_PU: ClassVar[float] = 1.8

    def __init__(self, p_kw: float, v_ll_v: float = 440.0, efficiency: float = 0.93,
                 pf_rated: float = 0.85, ilr_multiple: float = 6.5,
                 pf_locked: float = 0.3, s_sc_kva: float = 5000.0,
                 x_over_r: float = 5.0, t_acc_s: float = 8.0,
                 soft_limit: float = 3.0) -> None:
        """Crea el modelo validando todos los parámetros."""
        self.p_kw = require_range("Potencia", p_kw, 0.5, 5000.0)
        self.v_ll_v = require_range("Tensión", v_ll_v, 100.0, 15000.0)
        self.efficiency = require_range("Rendimiento", efficiency, 0.5, 1.0)
        self.pf_rated = require_range("FP nominal", pf_rated, 0.5, 1.0)
        self.ilr_multiple = require_range("I_arranque/I_n", ilr_multiple, 3.0, 9.0)
        self.pf_locked = require_range("FP rotor bloqueado", pf_locked, 0.1, 0.6)
        self.s_sc_kva = require_range("S_cc", s_sc_kva, 100.0, 1e7)
        self.x_over_r = require_range("X/R", x_over_r, 0.5, 30.0)
        self.t_acc_s = require_range("Tiempo de aceleración", t_acc_s, 0.5, 60.0)
        self.soft_limit = require_range("Límite arrancador suave", soft_limit, 1.5, 5.0)

    @property
    def i_rated_a(self) -> float:
        """Corriente nominal de línea [A]."""
        return self.p_kw * 1000 / (math.sqrt(3) * self.v_ll_v * self.efficiency * self.pf_rated)

    @property
    def s_rated_kva(self) -> float:
        """Potencia aparente nominal [kVA]."""
        return self.p_kw / (self.efficiency * self.pf_rated)

    def _shape(self, method: str, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Múltiplo de corriente (en I_n) y factor de potencia vs. tiempo."""
        if method not in self.METHODS:
            raise ParameterError(f"Método desconocido: {method}")
        s = np.exp(-3.0 * t / self.t_acc_s)
        direct = 1.0 + (self.ilr_multiple - 1.0) * s
        pf = self.pf_locked + (self.pf_rated - self.pf_locked) * (1.0 - s)
        if method == "Directo":
            mult = direct
        elif method == "Estrella-Triángulo":
            mult = np.where(t < 0.8 * self.t_acc_s, direct / 3.0, direct)
        elif method == "Arrancador suave":
            mult = np.minimum(direct, self.soft_limit)
        else:  # VFD
            mult, pf = np.full_like(t, 1.1), np.full_like(t, 0.95)
        return mult, pf

    def simulate(self, method: str, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simula tensión en el PCC y corriente de línea.

        Args:
            method: Uno de ``METHODS``.
            t: Vector de tiempo [s].

        Returns:
            Tupla (tensión [pu], corriente [A]).
        """
        mult, pf = self._shape(method, t)
        i_pu = mult * self.s_rated_kva / self.s_sc_kva          # base S_cc, Z_cc = 1 pu
        z_src = cmath.rect(1.0, math.atan(self.x_over_r))
        v = np.abs(1.0 / (1.0 + z_src * i_pu * np.exp(-1j * np.arccos(pf))))
        return v, mult * self.i_rated_a * v

    def start_torque_pu(self, method: str, v_min: float) -> float:
        """Par de arranque disponible [pu del par nominal] con la tensión v_min."""
        return self.LOCKED_TORQUE_PU * self.TORQUE_FACTOR[method] * v_min ** 2


class DailyLoadProfile:
    """Generador de curvas de carga diarias por estación."""

    # (base, pico mañana, pico mediodía, pico noche)
    SEASONS: ClassVar[Dict[str, Tuple[float, float, float, float]]] = {
        "Verano": (0.38, 0.25, 0.45, 0.80),
        "Invierno": (0.42, 0.40, 0.15, 1.00),
        "Primavera/Otoño": (0.35, 0.30, 0.20, 0.75),
    }

    @classmethod
    def generate(cls, season: str, p_peak_mw: float,
                 step_h: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve (horas, P [MW]) con el pico normalizado a ``p_peak_mw``."""
        if season not in cls.SEASONS:
            raise ParameterError(f"Estación desconocida: {season}")
        require_range("Potencia pico", p_peak_mw, 1.0, 1000.0)
        h = np.arange(0.0, 24.0, step_h)
        base, am, mid, night = cls.SEASONS[season]

        def gauss(mu: float, sig: float) -> np.ndarray:
            return np.exp(-0.5 * ((h - mu) / sig) ** 2)

        shape = base + am * gauss(8, 1.8) + mid * gauss(13, 2.5) + night * gauss(20, 2.0)
        return h, p_peak_mw * shape / shape.max()


def reactive_from_pf(p_mw: np.ndarray, pf: float, inductive: bool = True) -> np.ndarray:
    """Potencia reactiva [MVAr] de una carga con factor de potencia ``pf``."""
    require_range("Factor de potencia", pf, 0.5, 1.0)
    q = p_mw * math.tan(math.acos(pf))
    return q if inductive else -q


@dataclass(frozen=True)
class FeederModel:
    """Alimentador/subestación con carga PQ, banco de capacitores y OLTC.

    Resuelve V² - Vs·V + (R·P + X·Q_neta) = 0 de forma iterativa.

    Attributes:
        s_base_mva: Base de potencia [MVA].
        r_pu: Resistencia serie [pu].
        x_pu: Reactancia serie [pu].
        qc_pu: Carga capacitiva de la línea a 1 pu (efecto Ferranti) [pu].
        v_source_pu: Tensión de la fuente [pu].
    """

    s_base_mva: float = 100.0
    r_pu: float = 0.04
    x_pu: float = 0.18
    qc_pu: float = 0.12
    v_source_pu: float = 1.0

    def __post_init__(self) -> None:
        require_range("R", self.r_pu, 0.0, 1.0)
        require_range("X", self.x_pu, 0.0, 1.0)
        require_range("Qc línea", self.qc_pu, 0.0, 2.0)
        require_range("V fuente", self.v_source_pu, 0.9, 1.1)

    def receiving_voltage(self, p_mw: float, q_mvar: float, tap_steps: int = 0,
                          q_cap_mvar: float = 0.0) -> float:
        """Tensión en barra de carga [pu]; ``nan`` si hay colapso de tensión."""
        require_range("Tap", tap_steps, -8, 8)
        vs = self.v_source_pu * (1.0 + 0.0125 * tap_steps)
        p, q_load = p_mw / self.s_base_mva, q_mvar / self.s_base_mva
        q_shunt = q_cap_mvar / self.s_base_mva + self.qc_pu
        v = vs
        for _ in range(50):
            q_net = q_load - q_shunt * v * v
            disc = vs * vs - 4.0 * (self.r_pu * p + self.x_pu * q_net)
            if disc < 0:
                return float("nan")
            v_new = 0.5 * (vs + math.sqrt(disc))
            if abs(v_new - v) < 1e-7:
                return v_new
            v = v_new
        return v

    def profile(self, p_mw: np.ndarray, q_mvar: np.ndarray, tap_steps: int = 0,
                q_cap_mvar: np.ndarray | None = None) -> np.ndarray:
        """Perfil de tensión para series de P, Q y banco de capacitores."""
        caps = np.zeros_like(p_mw) if q_cap_mvar is None else q_cap_mvar
        return np.array([self.receiving_voltage(p, q, tap_steps, c)
                         for p, q, c in zip(p_mw, q_mvar, caps)])


@dataclass(frozen=True)
class FrequencyModel:
    """Respuesta frecuencia-potencia de un área (ecuación de oscilación).

    2H·dΔω/dt = ΔPm − ΔPL − D·Δω ;  Tg·dPv/dt = Pref − Δω/R − Pv ;
    Tt·dΔPm/dt = Pv − ΔPm ;  dPref/dt = −Ki·Δω (control secundario/AGC).

    Attributes:
        f0: Frecuencia nominal [Hz].
        s_base_mw: Base del sistema [MW].
        h: Constante de inercia [s].
        d: Autorregulación de carga [pu/pu].
        r: Estatismo del gobernador [pu].
        tg: Constante del gobernador [s].
        tt: Constante de la turbina [s].
        ki: Ganancia integral del AGC [pu/(pu·s)].
        reserve_mw: Reserva rodante disponible (límite de la válvula) [MW].
        governor_on: Activa la regulación primaria.
    """

    f0: float = 60.0
    s_base_mw: float = 1000.0
    h: float = 5.0
    d: float = 1.0
    r: float = 0.05
    tg: float = 0.5
    tt: float = 2.0
    ki: float = 0.0
    reserve_mw: float = 1000.0
    governor_on: bool = True

    def __post_init__(self) -> None:
        require_range("Inercia H", self.h, 1.0, 12.0)
        require_range("Amortiguamiento D", self.d, 0.0, 5.0)
        require_range("Estatismo R", self.r, 0.01, 0.12)
        require_range("Ki", self.ki, 0.0, 3.0)
        require_range("Reserva", self.reserve_mw, 0.0, self.s_base_mw)

    def _deriv(self, x: Sequence[float], dpl: float) -> List[float]:
        dw, pm, pv, pref = x
        droop = dw / self.r if self.governor_on else 0.0
        return [(pm - dpl - self.d * dw) / (2 * self.h),
                (pv - pm) / self.tt,
                (pref - droop - pv) / self.tg,
                -self.ki * dw]

    def simulate(self, delta_load_mw: float, t_event: float = 5.0,
                 t_end: float = 120.0, dt: float = 0.05) -> Dict[str, np.ndarray]:
        """Integra con RK4 un escalón de déficit (>0) o exceso (<0) de potencia.

        Returns:
            Diccionario con ``t`` [s], ``f`` [Hz], ``pm`` [MW] y ``pref`` [MW].
        """
        require_range("Magnitud del evento", abs(delta_load_mw), 0.0, self.s_base_mw)
        dpl_pu = delta_load_mw / self.s_base_mw
        lim = self.reserve_mw / self.s_base_mw
        n = int(t_end / dt) + 1
        t, out, x = np.arange(n) * dt, np.zeros((n, 4)), [0.0] * 4
        for k in range(n):
            out[k] = x
            dpl = dpl_pu if t[k] >= t_event else 0.0
            k1 = self._deriv(x, dpl)
            k2 = self._deriv([a + 0.5 * dt * b for a, b in zip(x, k1)], dpl)
            k3 = self._deriv([a + 0.5 * dt * b for a, b in zip(x, k2)], dpl)
            k4 = self._deriv([a + dt * b for a, b in zip(x, k3)], dpl)
            x = [a + dt / 6 * (b + 2 * c + 2 * e + g)
                 for a, b, c, e, g in zip(x, k1, k2, k3, k4)]
            x[2], x[3] = (min(max(v, -lim), lim) for v in (x[2], x[3]))  # saturación
        return {"t": t, "f": self.f0 * (1 + out[:, 0]),
                "pm": out[:, 1] * self.s_base_mw, "pref": out[:, 3] * self.s_base_mw}

    def rocof_hz_s(self, delta_load_mw: float) -> float:
        """Tasa inicial de cambio de frecuencia [Hz/s]."""
        return -(delta_load_mw / self.s_base_mw) * self.f0 / (2 * self.h)

    def static_deviation_hz(self, delta_load_mw: float) -> float:
        """Desviación de régimen permanente sin AGC [Hz]."""
        beta = self.d + (1.0 / self.r if self.governor_on else 0.0)
        return -self.f0 * (delta_load_mw / self.s_base_mw) / max(beta, 1e-9)


class EquipmentImpactModel:
    """Curvas didácticas de impacto de la tensión sobre equipos (Módulo 5)."""

    @staticmethod
    def motor_torque(v: float) -> float:
        """Par de un motor de inducción ∝ V² [pu]."""
        return v ** 2

    @staticmethod
    def motor_current(v: float) -> float:
        """Corriente a plena carga (potencia constante) ∝ 1/V [pu]."""
        return 1.0 / v

    @staticmethod
    def lamp_flux(v: float) -> float:
        """Flujo luminoso de lámpara incandescente ∝ V^3.4 [pu]."""
        return v ** 3.4

    @staticmethod
    def lamp_life(v: float) -> float:
        """Vida útil de lámpara incandescente ∝ V^-13 [pu]."""
        return v ** -13.0


class OverexcitationModel:
    """Relación V/f, protecciones de frecuencia y límites de turbina (Módulo 8)."""

    # (nombre, desplazamiento respecto a f0 [Hz], acción) - valores ilustrativos
    PROTECTION_STAGES: ClassVar[List[Tuple[str, float, str]]] = [
        ("81U-1", -0.7, "Deslastre de carga etapa 1"),
        ("81U-2", -1.2, "Deslastre de carga etapa 2"),
        ("81U-3", -1.8, "Deslastre etapa 3 / separación de islas"),
        ("81O-1", +1.0, "Reducción de generación"),
        ("81O-2", +1.5, "Disparo de unidad generadora"),
    ]

    def __init__(self, f0: float = 60.0) -> None:
        self.f0 = require_range("f0", f0, 45.0, 65.0)

    def vf_pu(self, v_pu: float, f_hz: float) -> float:
        """Relación V/f en por unidad."""
        require_range("Frecuencia", f_hz, 0.5 * self.f0, 1.5 * self.f0)
        return v_pu / (f_hz / self.f0)

    def triggered_stages(self, f_hz: float) -> List[str]:
        """Etapas 81U/81O cuyo ajuste se ha superado."""
        return [f"{n} ({a})" for n, off, a in self.PROTECTION_STAGES
                if (off < 0 and f_hz <= self.f0 + off) or (off > 0 and f_hz >= self.f0 + off)]

    def turbine_limit(self, f_hz: float) -> str:
        """Tiempo permitido de operación de turbina (curva ILUSTRATIVA)."""
        dev = abs(f_hz - self.f0) * 60.0 / self.f0
        for limit, text in ((0.5, "Continuo"), (1.0, "≈ 30 min acumulados"),
                            (1.5, "≈ 3 min"), (2.0, "≈ 30 s")):
            if dev <= limit:
                return text
        return "Disparo inmediato (resonancia de álabes)"

    def pump_power_pu(self, f_hz: float) -> float:
        """Potencia de bomba/ventilador ∝ n³ (leyes de afinidad) [pu]."""
        return (f_hz / self.f0) ** 3


# =============================================================================
# 3. CAPA DE INTERFAZ
# =============================================================================


@dataclass
class AppContext:
    """Contexto compartido por todos los módulos."""

    f0: float
    template: str
    dark: bool


@dataclass(frozen=True)
class Question:
    """Pregunta de opción múltiple del modo evaluación."""

    prompt: str
    options: Tuple[str, ...]
    answer: int
    explanation: str


def render_quiz(key: str, questions: Sequence[Question]) -> None:
    """Muestra preguntas con retroalimentación inmediata."""
    st.markdown("#### 🧪 Modo evaluación")
    for i, q in enumerate(questions):
        choice = st.radio(q.prompt, q.options, index=None, key=f"{key}_q{i}")
        if choice is not None:
            if q.options.index(choice) == q.answer:
                st.success("✅ Correcto. " + q.explanation)
            else:
                st.error("❌ Incorrecto. " + q.explanation)


def show(fig: go.Figure) -> None:
    """Dibuja una figura Plotly adaptada al ancho (compatible entre versiones)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def style_fig(fig: go.Figure, ctx: AppContext, title: str, xlab: str = "",
              ylab: str = "", height: int = 380) -> go.Figure:
    """Aplica estilo común (tema, leyenda, hover unificado, zoom)."""
    fig.update_layout(template=ctx.template, title=title, height=height,
                      margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.22))
    if xlab:
        fig.update_xaxes(title_text=xlab)
    if ylab:
        fig.update_yaxes(title_text=ylab)
    return fig


def kpi_row(items: Sequence[Tuple[str, str, str | None]]) -> None:
    """Fila de tarjetas métricas: (etiqueta, valor, delta opcional)."""
    for col, (label, value, delta) in zip(st.columns(len(items)), items):
        col.metric(label, value, delta, delta_color="off")


def md_table(rows: Sequence[Dict[str, object]]) -> None:
    """Tabla Markdown simple (evita depender de pandas para mostrar tablas)."""
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    st.markdown("\n".join(lines))


def inject_css(dark: bool) -> None:
    """Estilo industrial claro/oscuro.

    Fuerza fondo y texto en TODA la página (barra lateral, cabecera, listas
    desplegables y campos), de modo que el tema elegido no dependa del tema
    nativo del navegador/Streamlit.
    """
    bg, side, card, border, fg = (
        ("#0e141b", "#141c26", "#1b2430", "#2f3b4c", "#e6edf3") if dark
        else ("#f4f6f9", "#e6ebf1", "#ffffff", "#c9d2dd", "#1c2733"))
    st.markdown(f"""<style>
    html, body, .stApp, .stApp [data-testid="stAppViewContainer"],
    .stApp [data-testid="stMain"], .stApp .main {{ background:{bg} !important; }}
    .stApp header[data-testid="stHeader"], .stApp [data-testid="stToolbar"],
    .stApp [data-testid="stDecoration"] {{ background:{bg} !important; }}
    .stApp section[data-testid="stSidebar"],
    .stApp section[data-testid="stSidebar"] > div,
    .stApp [data-testid="stSidebarContent"] {{ background:{side} !important; }}
    .stApp, .stApp *, [data-baseweb="popover"] * {{ color:{fg} !important; }}
    .stApp svg, [data-baseweb="popover"] svg {{ fill:{fg}; }}
    .stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"],
    .stApp [data-baseweb="base-input"], [data-baseweb="popover"],
    [data-baseweb="popover"] ul, [data-baseweb="menu"],
    .stApp [data-testid="stNumberInput"] button,
    .stApp [data-testid="stExpander"] {{
        background:{card} !important; border-color:{border} !important; }}
    .stApp [data-testid="stMetric"] {{ background:{card} !important; border:1px solid {border};
        border-left:4px solid #f5a623; padding:12px 14px; border-radius:10px; }}
    .stApp [data-testid="stVerticalBlockBorderWrapper"] {{ border-color:{border}; }}
    .stApp table, .stApp th, .stApp td {{ border-color:{border} !important; }}
    </style>""", unsafe_allow_html=True)


class LabModule(ABC):
    """Clase base de un módulo del laboratorio.

    EXTENSIÓN: para añadir un tema nuevo cree una subclase, defina los
    atributos de clase y decórela con ``@register_module``. No hay que tocar
    ``main()`` ni los demás módulos.
    """

    module_id: ClassVar[str] = "base"
    title: ClassVar[str] = "Módulo"
    icon: ClassVar[str] = "⚡"
    objective: ClassVar[str] = ""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    def key(self, name: str) -> str:
        """Clave única de widget (evita colisiones entre pestañas)."""
        return f"{self.module_id}_{name}"

    @abstractmethod
    def render(self) -> None:
        """Dibuja el contenido del módulo."""

    def run(self) -> None:
        """Encabezado común + manejo robusto de errores."""
        st.header(f"{self.icon} {self.title}")
        if self.objective:
            st.caption(f"🎯 Objetivo: {self.objective}")
        try:
            self.render()
        except ParameterError as exc:
            st.error(f"Parámetro inválido: {exc}")
        except Exception as exc:  # noqa: BLE001 - la UI nunca debe caerse
            st.error(f"Error inesperado en el módulo: {exc}")


MODULE_REGISTRY: List[Type[LabModule]] = []


def register_module(cls: Type[LabModule]) -> Type[LabModule]:
    """Decorador que añade un módulo al laboratorio (orden de registro = pestañas)."""
    MODULE_REGISTRY.append(cls)
    return cls


# ----------------------------- Módulo 1 --------------------------------------
@register_module
class ControlImportanceModule(LabModule):
    """Importancia del control del sistema de potencia."""

    module_id, title, icon = "m1", "1. Control del SEP", "🎛️"
    objective = "Entender por qué P rige la frecuencia y Q rige la tensión."

    def render(self) -> None:
        c1, c2 = st.columns(2)
        c1.markdown("""**¿Por qué controlar?** La energía no se almacena en la red:
generación = demanda + pérdidas en todo instante. Si se rompe el balance, el sistema
reacciona con su energía cinética (inercia) y las protecciones actúan.""")
        with c2:
            md_table([
                {"Variable": "Frecuencia (f)", "Se controla con": "Potencia activa P",
                 "Naturaleza": "Global", "Actuadores": "Gobernadores, AGC, reserva"},
                {"Variable": "Tensión (V)", "Se controla con": "Potencia reactiva Q",
                 "Naturaleza": "Local", "Actuadores": "AVR, capacitores, taps, SVC"}])
        st.markdown("#### Panel de balance generación vs. demanda")
        a, b = st.columns(2)
        pg = a.slider("Generación programada P_g [MW]", 700, 1300, 1000, key=self.key("pg"))
        pd_ = a.slider("Demanda P_d [MW]", 700, 1300, 1000, key=self.key("pd"))
        qg = b.slider("Reactivo generado Q_g [MVAr]", 0, 600, 300, key=self.key("qg"))
        qd = b.slider("Reactivo demandado Q_d [MVAr]", 0, 600, 300, key=self.key("qd"))
        r_pct = a.slider("Estatismo R [%]", 1.0, 12.0, 5.0, 0.5, key=self.key("r")) / 100
        scc = b.slider("Potencia de cortocircuito [MVA]", 500, 10000, 3000, 100, key=self.key("scc"))
        model = PowerBalanceModel(f0=self.ctx.f0, droop=r_pct, s_sc_mva=float(scc))
        f, v = model.steady_frequency(pg, pd_), model.steady_voltage(qg, qd)
        kpi_row([("Frecuencia", f"{f:.3f} Hz", f"{f - self.ctx.f0:+.3f} Hz"),
                 ("Tensión", f"{v:.3f} pu", f"{v - 1:+.3f} pu"),
                 ("ΔP", f"{pg - pd_:+d} MW", None), ("ΔQ", f"{qg - qd:+d} MVAr", None)])
        if abs(f - self.ctx.f0) > 0.5:
            st.warning("Frecuencia fuera de ±0.5 Hz: riesgo de actuación de protecciones 81U/81O.")
        if not 0.95 <= v <= 1.05:
            st.warning("Tensión fuera de la banda 0.95–1.05 pu.")
        dp = np.linspace(-300, 300, 121)
        left, right = st.columns(2)
        fig = go.Figure(go.Scatter(x=dp, y=[model.steady_frequency(d + 1000, 1000) for d in dp],
                                   name="f(ΔP)"))
        fig.add_trace(go.Scatter(x=[pg - pd_], y=[f], mode="markers", name="Punto actual",
                                 marker=dict(size=12, color="orange")))
        with left:
            show(style_fig(fig, self.ctx, "Característica P–f", "ΔP = P_g − P_d [MW]", "f [Hz]"))
        dq = np.linspace(-300, 300, 121)
        fig2 = go.Figure(go.Scatter(x=dq, y=[model.steady_voltage(d, 0) for d in dq], name="V(ΔQ)"))
        fig2.add_trace(go.Scatter(x=[qg - qd], y=[v], mode="markers", name="Punto actual",
                                  marker=dict(size=12, color="orange")))
        with right:
            show(style_fig(fig2, self.ctx, "Característica Q–V", "ΔQ = Q_g − Q_d [MVAr]", "V [pu]"))
        render_quiz(self.module_id, [Question(
            "¿Qué variable refleja el balance GLOBAL de potencia activa?",
            ("La tensión", "La frecuencia", "El factor de potencia"), 1,
            "La frecuencia es común a todo el sistema síncrono; la tensión es local.")])


# ----------------------------- Módulo 2 --------------------------------------
@register_module
class MotorStartModule(LabModule):
    """Problemas de tensión durante el arranque de motores."""

    module_id, title, icon = "m2", "2. Arranque de motores", "⚙️"
    objective = "Cuantificar la caída de tensión e irrupción de corriente según el método."

    def render(self) -> None:
        a, b, c = st.columns(3)
        unit = a.radio("Unidad", ["kW", "HP"], horizontal=True, key=self.key("unit"))
        val = a.number_input(f"Potencia [{unit}]", 1.0, 3000.0, 75.0, key=self.key("p"))
        p_kw = val * (0.7457 if unit == "HP" else 1.0)
        v_ll = b.selectbox("Tensión de línea [V]", [208, 220, 380, 440, 480, 600], 3, key=self.key("v"))
        scc = b.slider("Potencia de cortocircuito S_cc [kVA]", 500, 50000, 3000, 100, key=self.key("scc"))
        t_acc = c.slider("Tiempo de aceleración [s]", 1.0, 30.0, 8.0, key=self.key("t"))
        load_t = c.slider("Par resistente [pu]", 0.0, 1.5, 0.5, 0.05, key=self.key("tl"))
        method = st.radio("Método de arranque", MotorStartingModel.METHODS, horizontal=True,
                          key=self.key("m"))
        model = MotorStartingModel(p_kw, float(v_ll), s_sc_kva=float(scc), t_acc_s=t_acc)
        t = np.linspace(0, 1.5 * t_acc, 300)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Tensión en el PCC", "Corriente de línea"))
        rows = []
        for m in MotorStartingModel.METHODS:
            v, i = model.simulate(m, t)
            w = 4 if m == method else 1.5
            fig.add_trace(go.Scatter(x=t, y=v, name=m, line=dict(width=w), legendgroup=m), 1, 1)
            fig.add_trace(go.Scatter(x=t, y=i, name=m, line=dict(width=w), legendgroup=m,
                                     showlegend=False), 2, 1)
            rows.append({"Método": m, "I_pico [A]": round(float(i.max()), 1),
                         "V_mín [pu]": round(float(v.min()), 3),
                         "Par arranque [pu]": round(model.start_torque_pu(m, float(v.min())), 2)})
        fig.add_hline(y=0.9, line_dash="dash", line_color="red", row=1, col=1,
                      annotation_text="Límite 0.90 pu")
        fig.update_xaxes(title_text="Tiempo [s]", row=2, col=1)
        fig.update_yaxes(title_text="V [pu]", row=1, col=1)
        fig.update_yaxes(title_text="I [A]", row=2, col=1)
        show(style_fig(fig, self.ctx, "Respuesta durante el arranque", height=560))
        sel = next(r for r in rows if r["Método"] == method)
        kpi_row([("I nominal", f"{model.i_rated_a:.1f} A", None),
                 ("I de arranque", f"{sel['I_pico [A]']:.0f} A",
                  f"{sel['I_pico [A]'] / model.i_rated_a:.1f} × I_n"),
                 ("Caída de tensión", f"{(1 - sel['V_mín [pu]']) * 100:.1f} %", None),
                 ("Par de arranque", f"{sel['Par arranque [pu]']:.2f} pu", None)])
        if sel["V_mín [pu]"] < 0.9:
            st.error("Caída > 10 %: riesgo de disparo de contactores, parpadeo (flicker) y fallo de arranque.")
        else:
            st.success("Caída de tensión admisible (≥ 0.90 pu).")
        if sel["Par arranque [pu]"] < load_t:
            st.error("El par disponible es menor que el par resistente: el motor NO arranca.")
        md_table(rows)
        render_quiz(self.module_id, [Question(
            "El arranque estrella-triángulo reduce la corriente de arranque aproximadamente a:",
            ("1/√3", "1/3", "1/9"), 1,
            "La tensión de fase baja a 1/√3, la corriente de fase a 1/√3 y la de línea a 1/3.")])


# ----------------------------- Módulo 3 --------------------------------------
@register_module
class LoadChangeModule(LabModule):
    """Problemas de tensión por cambios de carga en subestaciones."""

    module_id, title, icon = "m3", "3. Cambios de carga", "📈"
    objective = "Relacionar la curva de carga con la tensión (caída en pico, Ferranti en valle)."

    def render(self) -> None:
        a, b, c = st.columns(3)
        season = a.selectbox("Estación", list(DailyLoadProfile.SEASONS), key=self.key("s"))
        peak = a.slider("Demanda pico [MW]", 10, 150, 70, key=self.key("pk"))
        pf = b.slider("Factor de potencia", 0.70, 1.0, 0.90, 0.01, key=self.key("pf"))
        kind = b.radio("Tipo de carga", ["Inductiva", "Capacitiva"], horizontal=True, key=self.key("k"))
        qc = c.slider("Carga capacitiva de la línea [pu]", 0.0, 1.0, 0.25, 0.01, key=self.key("qc"))
        vs = c.slider("Tensión de la fuente [pu]", 0.95, 1.05, 1.0, 0.005, key=self.key("vs"))
        h, p = DailyLoadProfile.generate(season, float(peak))
        q = reactive_from_pf(p, pf, kind == "Inductiva")
        v = FeederModel(qc_pu=qc, v_source_pu=vs).profile(p, q)
        if np.isnan(v).any():
            st.error("Colapso de tensión: reduzca la demanda o mejore el factor de potencia.")
            return
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Curva de carga", "Tensión en la barra de carga"))
        fig.add_trace(go.Scatter(x=h, y=p, name="P [MW]", fill="tozeroy"), 1, 1)
        fig.add_trace(go.Scatter(x=h, y=q, name="Q [MVAr]"), 1, 1)
        fig.add_trace(go.Scatter(x=h, y=v, name="V [pu]", line=dict(color="orange")), 2, 1)
        fig.add_hrect(y0=0.95, y1=1.05, fillcolor="green", opacity=0.12, line_width=0, row=2, col=1)
        fig.update_xaxes(title_text="Hora del día [h]", row=2, col=1)
        fig.update_yaxes(title_text="MW / MVAr", row=1, col=1)
        fig.update_yaxes(title_text="V [pu]", row=2, col=1)
        show(style_fig(fig, self.ctx, f"Día típico de {season}", height=560))
        kpi_row([("V mínima", f"{v.min():.3f} pu", f"a las {h[v.argmin()]:.1f} h"),
                 ("V máxima", f"{v.max():.3f} pu", f"a las {h[v.argmax()]:.1f} h"),
                 ("Demanda mínima", f"{p.min():.1f} MW", None),
                 ("Horas fuera de banda", f"{np.mean((v < 0.95) | (v > 1.05)) * 24:.1f} h", None)])
        if v.max() > 1.05:
            st.warning("Sobretensión en horas valle: efecto Ferranti (carga capacitiva de la línea "
                       "o carga capacitiva con poca demanda inductiva que la compense).")
        if v.min() < 0.95:
            st.warning("Subtensión en horas pico: la corriente de carga aumenta la caída R·P + X·Q.")
        render_quiz(self.module_id, [Question(
            "¿Cuándo es más probable el efecto Ferranti en una línea larga?",
            ("En el pico de demanda", "En horas valle con poca carga", "Durante un cortocircuito"), 1,
            "Con poca carga, la corriente capacitiva de la línea sube la tensión en el extremo receptor.")])


# ----------------------------- Módulo 4 --------------------------------------
@register_module
class VoltagePracticeModule(LabModule):
    """Práctica interactiva: variaciones de tensión."""

    module_id, title, icon = "m4", "4. Práctica: tensión", "🛠️"
    objective = "Llevar el perfil de tensión de 24 h a la banda 0.95–1.05 pu con OLTC y capacitores."

    def render(self) -> None:
        st.info("**Caso:** alimentador largo en invierno (70 MW pico, FP 0.88). "
                "Reto: ≥ 95 % de las horas dentro de 0.95–1.05 pu sin horas fuera de 0.90–1.10 pu.")
        feeder = FeederModel(r_pu=0.04, x_pu=0.18, qc_pu=0.12)
        h, p = DailyLoadProfile.generate("Invierno", 70.0)
        q = reactive_from_pf(p, 0.88)
        left, right = st.columns([1, 2])
        with left:
            tap = st.slider("Tap OLTC (pasos de 1.25 %)", -8, 8, 0, key=self.key("tap"))
            steps = st.slider("Pasos de capacitor (5 MVAr c/u)", 0, 8, 0, key=self.key("n"))
            t_on = st.slider("Conectar banco a las [h]", 0, 23, 17, key=self.key("on"))
            t_off = st.slider("Desconectar banco a las [h]", 1, 24, 22, key=self.key("off"))
        mask = (h >= t_on) & (h < t_off) if t_on <= t_off else (h >= t_on) | (h < t_off)
        cap = np.where(mask, 5.0 * steps, 0.0)
        v0 = feeder.profile(p, q)
        v = feeder.profile(p, q, tap, cap)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h, y=v0, name="Sin control", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=h, y=v, name="Con control", line=dict(color="orange", width=3)))
        fig.add_hrect(y0=0.95, y1=1.05, fillcolor="green", opacity=0.12, line_width=0)
        with right:
            show(style_fig(fig, self.ctx, "Perfil de tensión de 24 h", "Hora [h]", "V [pu]"))
        if np.isnan(v).any():
            st.error("Colapso de tensión con esta configuración.")
            return
        ok_band = float(np.mean((v >= 0.95) & (v <= 1.05)))
        severe = bool(((v < 0.90) | (v > 1.10)).any())
        kpi_row([("Horas en banda", f"{ok_band * 100:.0f} %", None),
                 ("V mín / máx", f"{v.min():.3f} / {v.max():.3f} pu", None),
                 ("Q capacitiva", f"{5 * steps} MVAr", None)])
        if ok_band >= 0.95 and not severe:
            st.success("🏆 Reto superado: perfil dentro de norma.")
        else:
            if v.min() < 0.95:
                st.warning("Subtensión: suba el tap o conecte capacitores durante el pico.")
            if v.max() > 1.05:
                st.warning("Sobretensión en valle: baje el tap o desconecte el banco en horas valle.")
        render_quiz(self.module_id, [Question(
            "¿Qué efecto tiene dejar un banco de capacitores conectado todo el día?",
            ("Siempre mejora la tensión", "Puede causar sobretensión en el valle",
             "No afecta a la tensión"), 1,
            "En baja demanda el exceso de reactivo capacitivo eleva la tensión.")])


# ----------------------------- Módulo 5 --------------------------------------
@register_module
class VoltageEquipmentModule(LabModule):
    """Equipos afectados por cambios de tensión."""

    module_id, title, icon = "m5", "5. Equipos y tensión", "🔌"
    objective = "Ver cómo subtensión y sobretensión degradan transformadores, motores, luminarias y electrónica."

    def render(self) -> None:
        v = st.slider("Tensión de operación [pu]", 0.80, 1.20, 1.00, 0.01, key=self.key("v"))
        m = EquipmentImpactModel
        i_mag = magnetizing_current_pu(v) * 100
        if v < 0.85:
            elec = "⛔ Disparo por subtensión"
        elif v < 0.90:
            elec = "⚠️ Ride-through limitado"
        elif v <= 1.10:
            elec = "✅ Normal"
        else:
            elec = "⚠️ Estrés en bus DC / disparo OV"
        cards = [
            ("🧲 Transformador", f"I_mag = {i_mag:.1f} %",
             "↓V: sin riesgo mayor, regulación pobre.\n\n↑V: saturación del núcleo, calentamiento, armónicos."),
            ("⚙️ Motor de inducción", f"Par = {m.motor_torque(v) * 100:.0f} %  |  I = {m.motor_current(v) * 100:.0f} %",
             "↓V: par ∝ V², más deslizamiento y corriente → calentamiento.\n\n↑V: saturación, más pérdidas."),
            ("💡 Iluminación incandescente", f"Flujo = {m.lamp_flux(v) * 100:.0f} %  |  Vida = {min(m.lamp_life(v), 99) * 100:.0f} %",
             "↓V: luz muy reducida.\n\n↑V: vida útil se desploma (∝ V⁻¹³)."),
            ("🖥️ Electrónica de potencia", elec,
             "↓V: reinicios, pérdida de regulación del bus DC.\n\n↑V: sobretensión en semiconductores y capacitores."),
        ]
        for col, (name, value, text) in zip(st.columns(4), cards):
            with col.container(border=True):
                st.markdown(f"**{name}**")
                st.markdown(f"`{value}`")
                st.caption(text)
        vs = np.linspace(0.8, 1.2, 81)
        fig = go.Figure()
        for name, fn in (("Par del motor (V²)", m.motor_torque), ("Flujo lámpara (V^3.4)", m.lamp_flux),
                         ("Vida lámpara (V^-13)", m.lamp_life), ("Corriente motor (1/V)", m.motor_current),
                         ("I_mag transformador", lambda x: math.exp(22 * (x - 1)))):
            fig.add_trace(go.Scatter(x=vs, y=[fn(x) for x in vs], name=name))
        fig.add_vline(x=v, line_dash="dash", line_color="orange")
        fig.update_yaxes(type="log")
        show(style_fig(fig, self.ctx, "Impacto relativo (1.0 = valor a 1 pu)", "V [pu]", "Relativo (log)", 420))
        render_quiz(self.module_id, [Question(
            "Si la tensión cae al 90 %, el par del motor de inducción cae aproximadamente al:",
            ("90 %", "81 %", "73 %"), 1, "El par es proporcional a V²: 0.9² = 0.81.")])


# ----------------------------- Módulo 6 --------------------------------------
@register_module
class FrequencyProblemsModule(LabModule):
    """Problemas relacionados con la variación de frecuencia."""

    module_id, title, icon = "m6", "6. Variación de frecuencia", "〰️"
    objective = "Analizar la respuesta f(t) ante un desbalance usando la ecuación de oscilación."

    def render(self) -> None:
        a, b, c = st.columns(3)
        event = a.radio("Evento", ["Pérdida de generación", "Entrada masiva de demanda",
                                   "Rechazo de carga"], key=self.key("e"))
        size = a.slider("Magnitud [MW]", 10, 400, 150, key=self.key("mw"))
        h = b.slider("Inercia H [s]", 1.0, 10.0, 5.0, 0.5, key=self.key("h"))
        d = b.slider("Autorregulación D [pu/pu]", 0.0, 3.0, 1.0, 0.1, key=self.key("d"))
        r = c.slider("Estatismo R [%]", 1.0, 12.0, 5.0, 0.5, key=self.key("r")) / 100
        ref = c.checkbox("Mostrar caso sin gobernador (referencia)", key=self.key("ref"))
        delta = float(-size if event == "Rechazo de carga" else size)
        f0 = self.ctx.f0
        model = FrequencyModel(f0=f0, h=h, d=d, r=r)
        res = model.simulate(delta, t_end=60.0)
        fig = go.Figure(go.Scatter(x=res["t"], y=res["f"], name="Con gobernador",
                                   line=dict(width=3, color="orange")))
        if ref:
            r0 = FrequencyModel(f0=f0, h=h, d=d, r=r, governor_on=False).simulate(delta, t_end=60.0)
            fig.add_trace(go.Scatter(x=r0["t"], y=r0["f"], name="Sin gobernador", line=dict(dash="dot")))
        for name, off, _ in OverexcitationModel.PROTECTION_STAGES[:2] + OverexcitationModel.PROTECTION_STAGES[3:4]:
            fig.add_hline(y=f0 + off, line_dash="dash", line_color="red", annotation_text=name)
        show(style_fig(fig, self.ctx, "Frecuencia del sistema", "Tiempo [s]", "f [Hz]", 420))
        extreme = res["f"].min() if delta > 0 else res["f"].max()
        kpi_row([("RoCoF inicial", f"{model.rocof_hz_s(delta):+.3f} Hz/s", None),
                 ("Nadir/Cresta", f"{extreme:.3f} Hz", f"{extreme - f0:+.3f} Hz"),
                 ("f final (60 s)", f"{res['f'][-1]:.3f} Hz", None),
                 ("Δf estático teórico", f"{model.static_deviation_hz(delta):+.3f} Hz", None)])
        if abs(extreme - f0) >= 0.7:
            st.warning("Se supera el primer ajuste 81U/81O: habría deslastre/reducción automática.")
        render_quiz(self.module_id, [Question(
            "Si el sistema tiene MENOS inercia, ante la misma pérdida de generación:",
            ("El RoCoF es menor", "El RoCoF es mayor", "No cambia el RoCoF"), 1,
            "RoCoF = −ΔP·f0/(2H): a menor H, caída inicial más rápida.")])


# ----------------------------- Módulo 7 --------------------------------------
@register_module
class FrequencyPracticeModule(LabModule):
    """Práctica interactiva: regulación primaria y secundaria."""

    module_id, title, icon = "m7", "7. Práctica: frecuencia", "🎚️"
    objective = "Ajustar droop, AGC y reserva para recuperar la frecuencia nominal tras una contingencia."

    CONTINGENCIES: ClassVar[Dict[str, float]] = {
        "Disparo de unidad (60 MW)": 60.0, "Disparo de central (100 MW)": 100.0,
        "Pérdida de interconexión (150 MW)": 150.0}

    def render(self) -> None:
        f0 = self.ctx.f0
        st.info(f"**Metas:** (1) nadir > {f0 - 0.7:.1f} Hz (sin deslastre), (2) |f − {f0:.0f}| ≤ 0.05 Hz "
                "a los 150 s, (3) reserva suficiente para cubrir la contingencia.")
        ev = st.selectbox("Contingencia", list(self.CONTINGENCIES), key=self.key("ev"))
        a, b, c = st.columns(3)
        r = a.slider("Estatismo R [%] (control primario)", 1.0, 12.0, 5.0, 0.5, key=self.key("r")) / 100
        ki = b.slider("Ganancia AGC Ki (control secundario)", 0.0, 2.0, 0.0, 0.05, key=self.key("ki"))
        res_mw = c.slider("Reserva rodante [MW]", 0, 300, 200, 10, key=self.key("res"))
        delta = self.CONTINGENCIES[ev]
        model = FrequencyModel(f0=f0, h=5.0, r=r, ki=ki, reserve_mw=float(res_mw))
        out = model.simulate(delta, t_end=150.0)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Frecuencia", "Potencia mecánica añadida y referencia AGC"))
        fig.add_trace(go.Scatter(x=out["t"], y=out["f"], name="f [Hz]", line=dict(color="orange", width=3)), 1, 1)
        fig.add_hrect(y0=f0 - 0.05, y1=f0 + 0.05, fillcolor="green", opacity=0.15, line_width=0, row=1, col=1)
        fig.add_hline(y=f0 - 0.7, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_trace(go.Scatter(x=out["t"], y=out["pm"], name="ΔP_m [MW]"), 2, 1)
        fig.add_trace(go.Scatter(x=out["t"], y=out["pref"], name="P_ref AGC [MW]", line=dict(dash="dot")), 2, 1)
        fig.add_hline(y=delta, line_dash="dash", line_color="gray", row=2, col=1,
                      annotation_text="Contingencia")
        fig.update_xaxes(title_text="Tiempo [s]", row=2, col=1)
        fig.update_yaxes(title_text="f [Hz]", row=1, col=1)
        fig.update_yaxes(title_text="MW", row=2, col=1)
        show(style_fig(fig, self.ctx, "Resultado de la simulación", height=560))
        nadir, f_end = float(out["f"].min()), float(out["f"][-1])
        goals = {"Nadir sobre ajuste 81U": nadir > f0 - 0.7,
                 "Frecuencia restaurada (±0.05 Hz)": abs(f_end - f0) <= 0.05,
                 "Reserva suficiente": res_mw >= delta}
        kpi_row([("Nadir", f"{nadir:.3f} Hz", f"{nadir - f0:+.3f} Hz"),
                 ("f final", f"{f_end:.3f} Hz", f"{f_end - f0:+.3f} Hz"),
                 ("Metas cumplidas", f"{sum(goals.values())} / 3", None)])
        for name, ok in goals.items():
            st.write(("✅ " if ok else "❌ ") + name)
        if all(goals.values()):
            st.success("🏆 Reto superado: regulación primaria y secundaria bien coordinadas.")
        else:
            with st.expander("💡 Pista"):
                st.write("El droop solo limita la caída (queda error en régimen permanente); "
                         "el AGC (Ki) integra el error hasta devolver f a nominal, pero con Ki muy alto "
                         "aparecen oscilaciones. Si la reserva es menor que la contingencia, ninguna "
                         "ganancia restituye la frecuencia.")
        render_quiz(self.module_id, [Question(
            "¿Qué acción elimina el error de frecuencia remanente del control primario?",
            ("Aumentar la inercia", "Control secundario (AGC)", "Subir la tensión"), 1,
            "El AGC desplaza la referencia de carga-velocidad con acción integral hasta Δf = 0.")])


# ----------------------------- Módulo 8 --------------------------------------
@register_module
class FrequencyEquipmentModule(LabModule):
    """Equipos afectados por cambios de frecuencia."""

    module_id, title, icon = "m8", "8. Equipos y frecuencia", "🌀"
    objective = "Analizar sobreflujo V/f, protecciones 81 y efectos en turbinas, motores y bombas."

    def render(self) -> None:
        f0 = self.ctx.f0
        model = OverexcitationModel(f0)
        a, b = st.columns(2)
        v = a.slider("Tensión [pu]", 0.8, 1.3, 1.0, 0.01, key=self.key("v"))
        f = b.slider("Frecuencia [Hz]", f0 - 5.0, f0 + 5.0, f0, 0.05, key=self.key("f"))
        vf = model.vf_pu(v, f)
        rpm = 120 * f / 4
        kpi_row([("V/f", f"{vf:.3f} pu", "SOBREEXCITACIÓN" if vf > 1.1 else None),
                 ("I magnetizante", f"{magnetizing_current_pu(vf) * 100:.1f} %", None),
                 ("Velocidad motor 4 polos", f"{rpm:.0f} rpm", None),
                 ("Potencia bomba/ventilador", f"{model.pump_power_pu(f) * 100:.0f} %", None)])
        if vf > 1.1:
            st.error("V/f > 1.10 pu: saturación del núcleo, calentamiento y disparo de la protección 24 (V/Hz).")
        elif vf > 1.05:
            st.warning("V/f > 1.05 pu: límite continuo a plena carga (IEEE C57.12.00).")
        stages = model.triggered_stages(f)
        st.write("**Protecciones 81O/81U activadas:** " + (", ".join(stages) if stages else "ninguna"))
        st.write(f"**Tiempo permitido de turbina (ilustrativo):** {model.turbine_limit(f)}")
        fs = np.linspace(f0 - 5, f0 + 5, 200)
        fig = go.Figure(go.Scatter(x=fs, y=[model.vf_pu(v, x) for x in fs], name=f"V/f con V = {v:.2f} pu"))
        fig.add_hline(y=1.1, line_dash="dash", line_color="red", annotation_text="Límite 1.10")
        fig.add_hline(y=1.05, line_dash="dot", line_color="orange", annotation_text="1.05")
        fig.add_trace(go.Scatter(x=[f], y=[vf], mode="markers", name="Punto actual",
                                 marker=dict(size=12, color="orange")))
        show(style_fig(fig, self.ctx, "Relación V/f vs. frecuencia", "f [Hz]", "V/f [pu]"))
        st.markdown("**Ajustes ilustrativos de protección** (use los del código de red local):")
        md_table([{"Función": n, "Ajuste [Hz]": round(f0 + off, 2), "Acción": act}
                  for n, off, act in OverexcitationModel.PROTECTION_STAGES])
        render_quiz(self.module_id, [Question(
            "Un transformador a tensión nominal y frecuencia reducida 10 % queda con V/f ≈:",
            ("0.90 pu", "1.11 pu", "1.00 pu"), 1,
            "V/f = 1/0.9 ≈ 1.11 pu: sobreexcitación, el flujo es proporcional a V/f.")])


# =============================================================================
# 4. PUNTO DE ENTRADA
# =============================================================================


def main() -> None:
    """Ensambla el dashboard: barra lateral, tema y pestañas por módulo."""
    st.set_page_config(page_title="Laboratorio Virtual SEP", page_icon="⚡", layout="wide")
    with st.sidebar:
        st.title("⚡ Lab Virtual SEP")
        theme = st.radio("Tema", ["Oscuro industrial", "Claro"], key="theme")
        f0 = float(st.radio("Frecuencia nominal [Hz]", [60, 50], horizontal=True, key="f0"))
        st.caption("Modelos didácticos simplificados. Valores en por unidad (pu).")
        # EXTENSIÓN: añada aquí controles globales (p. ej., base de potencia del sistema).
    dark = theme.startswith("Oscuro")
    ctx = AppContext(f0=f0, template="plotly_dark" if dark else "plotly_white", dark=dark)
    inject_css(dark)
    st.title("Laboratorio Virtual de Sistemas Eléctricos de Potencia")
    # EXTENSIÓN: los módulos nuevos se registran con @register_module y aparecen aquí solos.
    modules = [cls(ctx) for cls in MODULE_REGISTRY]
    for tab, module in zip(st.tabs([f"{m.icon} {m.title}" for m in modules]), modules):
        with tab:
            module.run()


if __name__ == "__main__":
    main()

# Additional module clusters:
# using ${path.separator} (';' on Windows or ':' on Unix):
#
#netbeans_extraclusters="/absolute/path/to/cluster1:/absolute/path/to/cluster2"

