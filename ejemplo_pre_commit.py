"""Módulo de ejemplo clase Calculadora historial de operaciones."""

import json


class Calculadora:
    """Clase que realiza operaciones básicas y guarda un historial."""

    def __init__(self):
        """Inicializa una nueva instancia con historial vacío."""
        self.historial = []

    def sumar(self, a, b):
        """
        Suma dos números y guarda el resultado en el historial.

        Args:
            a (float): Primer número.
            b (float): Segundo número.

        Returns:
            float: Resultado de la suma.
        """
        resultado = a + b
        self.historial.append({"operacion": "suma", "resultado": resultado})
        return resultado

    def restar(self, a, b):
        """
        Resta dos números y guarda el resultado en el historial.

        Args:
            a (float): Minuendo.
            b (float): Sustraendo.

        Returns:
            float: Resultado de la resta.
        """
        resultado = a - b
        self.historial.append({"operacion": "resta", "resultado": resultado})
        return resultado

    def imprimir_historial(self):
        """Imprime el historial de operaciones realizadas."""
        for entrada in self.historial:
            print(entrada["operacion"], ":", entrada["resultado"])

    def guardar_historial(self, archivo):
        """
        Guarda el historial de operaciones en un archivo JSON.

        Args:
            archivo (str): Ruta del archivo donde se guardará el historial.
        """
        print("Guardando...")
        with open(archivo, "w") as f:
            json.dump(self.historial, f, indent=4)


def ejecutar():
    """Ejecuta un ejemplo de uso de la clase Calculadora."""
    print("Iniciando ejecución")
    calc = Calculadora()
    calc.sumar(5, 3)
    calc.restar(10, 2)
    calc.imprimir_historial()
    calc.guardar_historial("historial.json")


if __name__ == "__main__":
    ejecutar()
