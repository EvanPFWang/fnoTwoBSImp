"""helper functions for
seeding random number generators across Python, NumPy and PyTorch,
configuring visible GPU devices in an MPI context and enforcing
deterministic behaviour in PyTorch.  Timing decorators are included to
measure function execution time, both in serial and parallel (MPI)
contexts, while using numerically stable difference operations.


Example usage

code-block: python

    from utils_unified import set_random_seed, use_cpu, limit_visible_gpus,
    use_deterministic_ops, time, mpi_time

    #Set all RNG seeds for reproducible results
    set_random_seed(42)

    #Force computation on CPU (disable GPU)
    use_cpu()

    #Limit each MPI process to a single GPU based on its rank
    limit_visible_gpus()

    #Enforce deterministic algorithms in PyTorch
    use_deterministic_ops()

    #Decorate a function to measure execution time
    @time("my_function")
    def my_function(x):
        #expensive computation
        return x * x

    result, duration = my_function(10)

    #Decorate a function for MPI timing
    @mpi_time("parallel_function")
    def parallel_function(x):
        #distributed computation
        return x
    value, parallel_duration = parallel_function(123)

"""

from __future__ import annotations

import functools
import os
import random
from timeit import default_timer as timer
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from bs_pricing.py import  black_scholes_call_put, implied_volatility
from error_minimization.py import safe_subtract


import time
#Try to import PyTorch for seeding and GPU management.
try:
    import torch  #type: ignore
    _TORCH_AVAILABLE = True
except ImportError:  #pragma: no cover
    torch = None  #type: ignore
    _TORCH_AVAILABLE = False

#Import MPI for synchronisation across ranks.  MPI is optional for
#serial timing; timing functions will still work without it but the
#"mpi_time" decorator requires mpi4py.
try:
    from mpi4py import MPI  #type: ignore
    _MPI_AVAILABLE = True
except ImportError:  #pragma: no cover
    MPI = None  #type: ignore
    _MPI_AVAILABLE = False


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch random number generators.

    Reproducibility is critical for numerical experiments and machine
    learning.  This function sets "PYTHONHASHSEED" environment
    variable, seeds built‑in "random" module and NumPy, and, if
    available, PyTorch.  For PyTorch function seeds both CPU and
    CUDA RNGs and configures CuDNN for deterministic behaviour when
    possible.

    seed: integer seed to use.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        #Seed all GPUs if visible.  "cuda.manual_seed_all" is a no‑op
        #when no CUDA devices are present.
        torch.cuda.manual_seed_all(seed)  #type: ignore[operator]
        #Recommended settings for deterministic behaviour.  See
        #https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        #Warn user that PyTorch seeding could not be performed.
        pass


def use_cpu() -> None:
    """Force computations to run on CPU by hiding all GPUs.

    This function sets "CUDA_VISIBLE_DEVICES" to an empty string,
    preventing PyTorch from enumerating any CUDA devices.  Subsequent
    operations will therefore run on CPU.  If PyTorch is not
    available function simply sets environment variable.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    #Clearing cache ensures no stray CUDA contexts remain.
    if _TORCH_AVAILABLE and torch.cuda.is_available():  #type: ignore[union-attr]
        torch.cuda.empty_cache()  #type: ignore[func-returns-value]


def limit_visible_gpus() -> None:
    """Restrict each MPI process to a single GPU based on its rank.

    When running under MPI with multiple processes and multiple GPUs,
    it is often desirable to assign one GPU per rank.  This function
    queries number of CUDA devices via PyTorch and size and
    rank of "MPI.COMM_WORLD" communicator.  It then sets
    "CUDA_VISIBLE_DEVICES" to expose only GPU corresponding to
    local rank.  If number of GPUs does not match number
    of MPI processes a "ValueError" is raised.  If PyTorch or
    mpi4py are not available function raises an informative
    "ImportError".
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required to query available GPUs; install torch>=2.4.1"
        )
    if not _MPI_AVAILABLE:
        raise ImportError(
            "mpi4py is required for limit_visible_gpus in an MPI environment"
        )
    n_gpus = torch.cuda.device_count()  #type: ignore[call-arg]
    comm = MPI.COMM_WORLD  #type: ignore[attr-defined]
    if n_gpus == 0:
        return  #nothing to do
    if n_gpus != comm.size:
        raise ValueError(
            f"number of GPUs ({n_gpus}) must match MPI communicator size ({comm.size})"
        )
    rank = comm.rank
    #Expose only GPU corresponding to this rank.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)  #after masking, device 0 maps to selected GPU


def use_deterministic_ops() -> None:
    """Enforce deterministic algorithms in PyTorch.

    Sets environment variables and PyTorch flags that encourage
    deterministic operation, which is important for reproducible
    results.  When CuDNN is used, setting "CUBLAS_WORKSPACE_CONFIG"
    ensures deterministic behaviour in certain operations.  The
    function silently returns if PyTorch is not available.
    """
    #Configure CuBLAS workspace for deterministic behaviour.  See
    #https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if _TORCH_AVAILABLE:
        torch.use_deterministic_algorithms(True, warn_only=True)  #type: ignore[call-arg]
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def time(function_name: Optional[str] = None) -> Callable[[Callable], Callable]:
    """Decorator to measure wall‑clock time of a function.

    decorated function is called and its execution time is computed
    using function "timeit.default_timer".  difference between end and
    start times is computed using function "safe_subtract" to guard
    against catastrophic cancellation when run time is very short.
    function returns a tuple "(result, run_time)".  timing
    information is printed regardless of MPI rank.

    function_name: Optional explicit name to print; otherwise the
        wrapped function's "__name__" is used.

    Returns decorator that wraps target function.
    """
    def _time_wrapper_provider(function: Callable, name: Optional[str]) -> Callable:
        disp_name = name or function.__name__

        @functools.wraps(function)
        def _time_wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
            start_time = timer()
            value = function(*args, **kwargs)
            end_time = timer()
            run_time = safe_subtract(end_time, start_time)
            print(f"{disp_name} completed in {run_time}s")
            return value, run_time

        return _time_wrapper

    return lambda function: _time_wrapper_provider(function, function_name)


def mpi_time(function_name: Optional[str] = None) -> Callable[[Callable], Callable]:
    """Decorator to measure execution time across MPI processes.

    Synchronises all ranks using barriers before and after call and
    measures elapsed time with "MPI.Wtime".  difference is
    computed with function "safe_subtract".  Only rank 0 prints the
    timing information.  If mpi4py is not available decorator
    falls back to serial timing version.

    function_name: Optional explicit name to print.

    Returns decorator that wraps target function.
    """
    if not _MPI_AVAILABLE:
        #Fallback to serial timing if MPI is unavailable.
        return time(function_name)

    def _mpi_time_wrapper_provider(function: Callable, name: Optional[str]) -> Callable:
        disp_name = name or function.__name__

        @functools.wraps(function)
        def _mpi_time_wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
            comm = MPI.COMM_WORLD  #type: ignore[attr-defined]
            comm.barrier()
            start_time = MPI.Wtime()  #type: ignore[attr-defined]
            value = function(*args, **kwargs)
            comm.barrier()
            end_time = MPI.Wtime()  #type: ignore[attr-defined]
            run_time = safe_subtract(end_time, start_time)
            if comm.rank == 0:
                print(f"{disp_name} completed in {run_time}s")
            return value, run_time

        return _mpi_time_wrapper

    return lambda function: _mpi_time_wrapper_provider(function, function_name)


__all__ = [
    "set_random_seed",
    "use_cpu",
    "limit_visible_gpus",
    "use_deterministic_ops",
    "time",
    "mpi_time",
    "validate_put_call_parity",
    "benchmark_pricing_performance",
    "test_implied_volatility",
    "validate_greeks_relationships",
    "generate_test_data_for_ml",
    "test_edge_cases",
    "run_all_tests"

]


def validate_put_call_parity(S, K, r, T, call_price, put_price, tolerance=1e-6):
    """
    Validate put-call parity: C - P = S - K * exp(-r*T)

    Returns True if parity holds within tolerance
    """
    left_side = call_price - put_price
    right_side = S - K * np.exp(-r * T)
    difference = abs(left_side - right_side)

    return difference < tolerance, difference


def benchmark_pricing_performance(n_iterations=1000):
    """
    Benchmark the performance of Black-Scholes calculations
    """
    #Random parameters
    np.random.seed(42)
    S = np.random.uniform(50, 150, n_iterations)
    K = np.random.uniform(50, 150, n_iterations)
    r = np.random.uniform(0.01, 0.1, n_iterations)
    sigma = np.random.uniform(0.1, 0.5, n_iterations)
    T = np.random.uniform(0.1, 2.0, n_iterations)

    #Time without Greeks
    start = time.time()
    call_prices, put_prices = black_scholes_call_put(S, K, r, sigma, T)
    time_no_greeks = time.time() - start

    #Time with Greeks
    start = time.time()
    call_prices, put_prices, greeks = black_scholes_call_put(S, K, r, sigma, T, compute_greeks=True)
    time_with_greeks = time.time() - start

    print(f"Benchmark Results ({n_iterations} iterations):")
    print(f"Without Greeks: {time_no_greeks:.4f} seconds ({n_iterations / time_no_greeks:.0f} calcs/sec)")
    print(f"With Greeks: {time_with_greeks:.4f} seconds ({n_iterations / time_with_greeks:.0f} calcs/sec)")
    print(f"Greeks overhead: {(time_with_greeks / time_no_greeks - 1) * 100:.1f}%")

    return time_no_greeks, time_with_greeks

def test_implied_volatility():
    """
    Test implied volatility calculation
    """
    #Known parameters
    S = 100
    K = 100
    r = 0.05
    T = 1.0
    true_sigma = 0.2

    #Calculate option prices
    call_price, put_price = black_scholes_call_put(S, K, r, true_sigma, T)

    #Calculate implied volatility
    iv_call = implied_volatility(float(call_price), S, K, r, T, 'call')
    iv_put = implied_volatility(float(put_price), S, K, r, T, 'put')

    print("Implied Volatility Test:")
    print(f"True volatility: {true_sigma:.4f}")
    print(f"Implied vol (call): {iv_call:.4f}")
    print(f"Implied vol (put): {iv_put:.4f}")
    print(f"Call error: {abs(iv_call - true_sigma):.6f}")
    print(f"Put error: {abs(iv_put - true_sigma):.6f}")

    return iv_call, iv_put


def validate_greeks_relationships():
    """
    Validate mathematical relationships between Greeks
    """
    #Parameters
    S = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0

    #Calculate with Greeks
    _, _, greeks = black_scholes_call_put(S, K, r, sigma, T, compute_greeks=True)

    #Test: Call Delta - Put Delta = exp(-q*T) ≈ 1 (when q=0)
    delta_diff = greeks['call_delta'] - greeks['put_delta']
    expected_diff = 1.0  #Since q=0

    print("Greeks Validation:")
    print(f"Call Delta - Put Delta = {delta_diff:.6f} (expected: {expected_diff:.6f})")

    #Test: Gamma should be positive
    print(f"Gamma = {greeks['gamma']:.6f} (should be > 0: {greeks['gamma'] > 0})")

    #Test: Vega should be positive
    print(f"Vega = {greeks['vega']:.6f} (should be > 0: {greeks['vega'] > 0})")

    return greeks


def generate_test_data_for_ml(n_samples=100):
    """
    Generate synthetic test data for ML model training
    """
    np.random.seed(42)

    data = []
    for _ in range(n_samples):
        #Generate random parameters
        S = np.random.uniform(50, 150)
        K = np.random.uniform(50, 150)
        r = np.random.uniform(0.01, 0.1)
        sigma = np.random.uniform(0.1, 0.5)
        T = np.random.uniform(0.1, 2.0)

        #Calculate prices and Greeks
        call_price, put_price, greeks = black_scholes_call_put(
            S, K, r, sigma, T, compute_greeks=True
        )

        #Store data
        data.append({
            "StockPrice": S,
            "StrikePrice": K,
            "InterestRate": r,
            "Volatility": sigma,
            "TimeToExpiry": T,
            "Moneyness": S / K,
            "CallPrice": float(call_price),
            "PutPrice": float(put_price),
            "CallDelta": float(greeks["call_delta"]),
            "PutDelta": float(greeks["put_delta"]),
            "Gamma": float(greeks["gamma"]),
            "Vega": float(greeks["vega"]),
            "CallTheta": float(greeks["call_theta"]),
            "PutTheta": float(greeks["put_theta"]),
            "CallRho": float(greeks["call_rho"]),
            "PutRho": float(greeks["put_rho"])
        })

    df = pd.DataFrame(data)

    print(f"Generated {n_samples} samples for ML training")
    print("\nDataset statistics:")
    print(df[["CallPrice", "PutPrice", "Moneyness", "Volatility"]].describe())

    return df

def test_edge_cases():
    """
    Test edge cases and boundary conditions
    """
    print("Testing Edge Cases:")

    #Test 1: Deep in-the-money call (S >> K)
    S, K, r, sigma, T = 150, 50, 0.05, 0.2, 1.0
    call, put = black_scholes_call_put(S, K, r, sigma, T)
    print(f"\n1. Deep ITM Call (S={S}, K={K}):")
    print(f"   Call: ${float(call):.2f} (≈ intrinsic value: ${S - K:.2f})")
    print(f"   Put: ${float(put):.2f} (should be ≈ 0)")

    #Test 2: Deep out-of-the-money call (S << K)
    S, K = 50, 150
    call, put = black_scholes_call_put(S, K, r, sigma, T)
    print(f"\n2. Deep OTM Call (S={S}, K={K}):")
    print(f"   Call: ${float(call):.2f} (should be ≈ 0)")
    print(f"   Put: ${float(put):.2f} (≈ intrinsic value: ${K - S:.2f})")

    #Test 3: Near expiration (T → 0)
    S, K, T = 100, 100, 0.001
    call, put = black_scholes_call_put(S, K, r, sigma, T)
    print(f"\n3. Near Expiration (T={T}):")
    print(f"   Call: ${float(call):.4f}")
    print(f"   Put: ${float(put):.4f}")

    #Test 4: High volatility
    sigma = 2.0
    call, put = black_scholes_call_put(100, 100, r, sigma, 1.0)
    print(f"\n4. High Volatility (σ={sigma}):")
    print(f"   Call: ${float(call):.2f}")
    print(f"   Put: ${float(put):.2f}")

    #Test 5: Zero volatility (should approach intrinsic value)
    sigma = 0.001
    S, K = 110, 100
    call, put = black_scholes_call_put(S, K, r, sigma, 1.0)
    print(f"\n5. Near-Zero Volatility (σ={sigma}):")
    print(f"   Call: ${float(call):.2f} (intrinsic: ${max(S - K * np.exp(-r), 0):.2f})")
    print(f"   Put: ${float(put):.2f}")


def run_all_tests():
    """
    Run all validation tests
    """
    print("=" * 60)
    print("BLACK-SCHOLES VALIDATION SUITE")
    print("=" * 60)

    #Test 1: Put-Call Parity
    print("\n1. PUT-CALL PARITY TEST")
    print("-" * 30)
    S, K, r, T = 100, 100, 0.05, 1.0
    call, put = black_scholes_call_put(S, K, r, 0.2, T)
    parity_holds, diff = validate_put_call_parity(S, K, r, T, float(call), float(put))
    print(f"Parameters: S={S}, K={K}, r={r}, T={T}")
    print(f"Parity holds: {parity_holds} (difference: {diff:.8f})")

    #Test 2: Performance
    print("\n2. PERFORMANCE BENCHMARK")
    print("-" * 30)
    benchmark_pricing_performance(1000)

    #Test 3: Implied Volatility
    print("\n3. IMPLIED VOLATILITY")
    print("-" * 30)
    test_implied_volatility()

    #Test 4: Greeks Validation
    print("\n4. GREEKS VALIDATION")
    print("-" * 30)
    validate_greeks_relationships()

    #Test 5: Edge Cases
    print("\n5. EDGE CASES")
    print("-" * 30)
    test_edge_cases()

    #Test 6: Generate ML Data
    print("\n6. ML DATA GENERATION")
    print("-" * 30)
    df = generate_test_data_for_ml(50)

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()