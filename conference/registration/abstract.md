

# Abstract

One of the most common methods for synthesizing Yttrium Aluminium Garnet (YAG) crystals is solid-state reaction, which involves heating a homogeneous mixture of fine aluminum and yttrium oxides. It is known that the synthesis rate can be increased by periodically mixing the reagents. While some studies have investigated mathematical models describing this reaction, comprehensive research on the effects of mixing remains limited.

In this paper, a two-dimensional computational model was implemented using an explicit finite difference scheme based on the YAG diffusion-reaction system proposed by Ivanauskas et al. <cite>[1]</cite>:

$$
\begin{cases}
\dot{c_1}=D\Delta c_1-3kc_1c_2\\
\dot{c_2}=D\Delta c_2-5kc_1c_2\\
\dot{c_3}=2kc_1c_2
\end{cases}
$$

A mathematical analysis demonstrated that the time step required for numerical stability depends on the diffusion and reaction rate constants, the initial reagent concentrations, as well as the discrete spatial step sizes:

$$
\Delta t \leqslant (15kc_0+2D((\Delta x)^{-2}+(\Delta y)^{-2}))^{-1}
$$

The mixing of reagents was modeled as an instantaneous process occurring between a chosen discrete time step $t_n$ and its successor $t_{n+1}$ ​. Since the reaction area was simulated at a microscopic scale to ensure consistency with experimentally measured diffusion and reaction rate constants, individual reagent particles were represented at high spatial resolution. Given that mixing in solid-state reactions does not break down powders into finer particles, the process was modeled as a rearrangement of existing particles rather than fragmentation.

Two distinct mixing models were proposed: (1) Random mixing, where particles are shuffled and rotated unpredictably, and (2) Optimal mixing, where particles are rearranged to maximize reaction speed. Both models were implemented and integrated into the computational YAG reaction model, and the resulting effects were analyzed and compared.

# Annotation

Neodymium-doped yttrium aluminium garnet (YAG) crystals are a widely used material for active laser media. One of the most common methods for synthesizing Yttrium Aluminium Garnet (YAG) crystals is solid-state reaction, which involves heating a homogeneous mixture of fine aluminum and yttrium oxides. It is known that the synthesis rate can be increased by periodically mixing the reagents. While some studies have investigated mathematical models describing this reaction, comprehensive research on the effects of mixing remains limited.

In this research, a two-dimensional computational model was implemented using an explicit finite difference scheme based on the YAG diffusion-reaction system proposed by Ivanauskas et al. The mixing of reagents was modeled as an instantaneous process occurring between discrete time steps​. Since the reaction area was simulated at a microscopic scale to ensure consistency with experimentally measured diffusion and reaction rate constants, individual reagent particles were represented at high spatial resolution. Given that mixing in solid-state reactions does not break down powders into finer particles, the process was modeled as a rearrangement of existing particles rather than fragmentation. 

Two distinct mixing models were proposed: (1) Random mixing, where particles are shuffled and rotated unpredictably, and (2) Optimal mixing, where particles are rearranged to maximize reaction speed. Both models were implemented and integrated into the computational YAG reaction model. It was observed that the optimal mixing model more accurately represents the acceleration in reaction time observed in practice. Additionally, optimal mixing exhibits similar behavior in a larger reaction space.