
import sys
from bs_pricing import black_scholes_call_put
import numpy as np

def prompt_float(msg, default=None):
    while True:
        try:
            s = input(f"{msg}" + (f" [{default}]" if default is not None else "") + ": ").strip()
            if s == "" and default is not None:
                return float(default)
            return float(s)
        except ValueError:
            print("Please enter a number.")

def main():
    print("Black–Scholes Pricer (European options)")
    S = prompt_float("Spot price S", 100.0)
    K = prompt_float("Strike K", 100.0)
    r = prompt_float("Risk-free rate r (e.g., 0.05 for 5%)", 0.05)
    sigma = prompt_float("Volatility sigma (e.g., 0.2 for 20%)", 0.2)
    T = prompt_float("Time to expiry T in years (e.g., 0.5 for 6 months)", 1.0)

    call, put = black_scholes_call_put(S, K, r, sigma, T)
    print("\nResults")
    print("-------")
    print(f"Call: {float(call):.6f}")
    print(f" Put: {float(put):.6f}")

if __name__ == "__main__":
    main()
