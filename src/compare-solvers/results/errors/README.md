# Komentarai dėl rezultatų

- Rezultatuose vaizduojamas "point-wise relative error", kuris skaičiuojamas tokiu principu: $|q_{1i} - q_{2i}| / |q_{1i}|$
- Visų rezoliucijų sprendiniai gauti su tuo pačiu laiko žingsniu $\Delta t = 1s$
- "Tikrasis" sprendinys, su kuriuo lyginami kiti laikomas to pačio _**ADI**_ metodo sprendinys, kurio rezoliucija yra $200\times200$
- Trečiai medžiagai santykinė klaida atrodo didelė, nes pačios medžiagos kiekis reakcijos pradžioje yra $0$

## L normos

Absoliutaus sprendinių skirtumo $\textbf{q}_{200\times200}-\textbf{q}_{S\times S}, S\in\{40, 60, 80, 120\}$ normos:

### Pirma medžiaga $(c_1)$

| Rezoliucija      | $L^2$-norm            | $L^{\infty}$-norm     |
| ---------------- | --------------------- | --------------------- |
| $40 \times 40$   | $9.47\mathrm{e}{-08}$ | $1.07\mathrm{e}{-08}$ |
| $60 \times 60$   | $5.41\mathrm{e}{-08}$ | $6.16\mathrm{e}{-09}$ |
| $80 \times 80$   | $3.44\mathrm{e}{-08}$ | $3.92\mathrm{e}{-09}$ |
| $120 \times 120$ | $1.51\mathrm{e}{-08}$ | $1.73\mathrm{e}{-09}$ |

### Antra medžiaga $(c_2)$

| Rezoliucija      | $L^2$-norm            | $L^{\infty}$-norm     |
| ---------------- | --------------------- | --------------------- |
| $40 \times 40$   | $1.58\mathrm{e}{-07}$ | $1.79\mathrm{e}{-08}$ |
| $60 \times 60$   | $9.04\mathrm{e}{-08}$ | $1.03\mathrm{e}{-08}$ |
| $80 \times 80$   | $5.75\mathrm{e}{-08}$ | $6.55\mathrm{e}{-09}$ |
| $120 \times 120$ | $2.53\mathrm{e}{-08}$ | $2.89\mathrm{e}{-09}$ |


### Trečia medžiaga $(c_3)$

| Rezoliucija      | $L^2$-norm            | $L^{\infty}$-norm     |
| ---------------- | --------------------- | --------------------- |
| $40 \times 40$   | $6.32\mathrm{e}{-08}$ | $7.16\mathrm{e}{-09}$ |
| $60 \times 60$   | $3.61\mathrm{e}{-08}$ | $4.10\mathrm{e}{-09}$ |
| $80 \times 80$   | $2.29\mathrm{e}{-08}$ | $2.62\mathrm{e}{-09}$ |
| $120 \times 120$ | $1.01\mathrm{e}{-08}$ | $1.15\mathrm{e}{-09}$ |