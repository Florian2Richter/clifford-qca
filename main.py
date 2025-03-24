#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Define basic 2x2 matrices and the two-qubit CNOT gate (control = first qubit)
I = np.array([[1, 0],
              [0, 1]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1,  1],
                               [1, -1]], dtype=complex)

# CNOT acting on 2 qubits (control = qubit 0, target = qubit 1)
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)

def initial_state(N):
    """
    Initialize an N-qubit state with each qubit in the |+> state.
    |+> = H|0> = (|0> + |1>)/sqrt(2)
    """
    plus = (1/np.sqrt(2)) * np.array([1, 1], dtype=complex)
    state = plus
    for _ in range(N - 1):
        state = np.kron(state, plus)
    return state

def apply_two_qubit_gate(state, gate, control, target, N):
    """
    Apply a two-qubit gate (e.g., CNOT) to qubits at positions 'control' and 'target'
    in an N-qubit system by iterating over the computational basis.
    This method is simple and works for small systems.
    """
    dim = 2 ** N
    new_state = np.zeros(dim, dtype=complex)
    for i in range(dim):
        # Get binary representation of basis index as a list of bits
        bits = [(i >> j) & 1 for j in range(N)]
        # Determine the 2-bit sub-index for the controlled gate
        sub_index = bits[control] * 2 + bits[target]
        # The gate maps the two-bit state to a superposition of two-bit states.
        for new_sub_index in range(4):
            amplitude = gate[sub_index, new_sub_index]
            if abs(amplitude) > 1e-12:
                new_bits = bits.copy()
                new_bits[control] = (new_sub_index >> 1) & 1
                new_bits[target] = new_sub_index & 1
                # Convert new_bits back into an integer index
                new_index = sum([bit << j for j, bit in enumerate(new_bits)])
                new_state[new_index] += amplitude * state[i]
    return new_state

def simulate_qca(N, T):
    """
    Simulate the time evolution of a 1D Clifford QCA.
    N: number of qubits.
    T: number of time steps.
    
    For even time steps, apply CNOT gates on pairs (0,1), (2,3), ...
    For odd time steps, apply CNOT gates on pairs (1,2), (3,4), ...
    """
    state = initial_state(N)
    states = [state.copy()]
    
    for t in range(T):
        # Even time step: operate on pairs starting with index 0
        if t % 2 == 0:
            for i in range(0, N - 1, 2):
                state = apply_two_qubit_gate(state, CNOT, i, i + 1, N)
        # Odd time step: operate on pairs starting with index 1
        else:
            for i in range(1, N - 1, 2):
                state = apply_two_qubit_gate(state, CNOT, i, i + 1, N)
        states.append(state.copy())
    return states

def plot_probability_distribution(state, N, title="Probability Distribution"):
    """
    Plot the probability distribution of the computational basis states.
    """
    dim = 2 ** N
    probs = np.abs(state) ** 2
    plt.figure()
    plt.bar(range(dim), probs)
    plt.xlabel("Basis state (integer representation)")
    plt.ylabel("Probability")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    N = 4    # number of qubits (small number due to exponential scaling)
    T = 10   # total time steps to simulate
    states = simulate_qca(N, T)
    
    # Plot the probability distribution of the final state
    plot_probability_distribution(states[-1], N, title="Final State Distribution")
    
    # Optionally, print probability distributions at each time step
    for t, state in enumerate(states):
        print(f"Time step {t}:")
        print(np.round(np.abs(state)**2, 3))
