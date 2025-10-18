"""
Utilidades para manejo de fechas y campañas.
"""
from datetime import datetime, timedelta
import pandas as pd
from config.config import MES_INICIO_CAMPAÑA


class FechaUtils:
    """Utilidades para cálculos con fechas."""
    
    @staticmethod
    def obtener_lunes_semana(fecha):
        """
        Obtiene el lunes de la semana de una fecha dada.
        """
        return (fecha - timedelta(days=fecha.weekday())).date()
    
    @staticmethod
    def calcular_campaña(fecha):
        """
        Calcula la campaña (formato YY-YY) para una fecha dada.
        """
        año = fecha.year
        mes = fecha.month
        
        if mes < MES_INICIO_CAMPAÑA:
            return f"{str(año-2001)}-{str(año-2000)}"
        else:
            return f"{str(año-2000)}-{str(año-1999)}"
    
    @staticmethod
    def calcular_semana_campaña(fecha, inicio_campaña):
        """
        Calcula el número de semana dentro de una campaña.
        """
        dias_diferencia = (fecha - inicio_campaña).days
        return (dias_diferencia // 7) + 1