# Guia simple del flujo experimental

Esta guia explica, en pasos cortos, como corre un experimento con `tarware_ext` y `scripts/eval.py`.

## 1) Capas principales

- **scripts/**: puntos de entrada (CLI). Aqui eliges env, policy y parametros.
- **tarware_ext/**: capa experimental (runner, adapter, metrics, policies).
- **tarware/**: simulador base (env Gym + heuristica original).

## 2) Flujo basico (paso a paso)

1. `scripts/eval.py` crea el entorno con `gym.make(env_id)`.
2. Se selecciona la policy (random, heuristic, graph_greedy).
3. El runner ejecuta episodios y pasos.
4. El adapter normaliza `reset()` y `step()` en un `Transition` consistente.
5. La policy produce acciones (episodica o step-wise).
6. Metrics calcula resumen y el logger escribe el CSV.

## 3) Parametros clave

- `--env-id`: define el entorno (tamano, agentes, obs).
- `--policy`: random | heuristic | graph_greedy.
- `--episodes`, `--steps`, `--seed`: control del experimento.
- `--distance`: `manhattan` o `find_path` (solo graph_greedy).
- `--active-alpha`: limita AGVs activos. Regla base: `max_active_agvs = active_alpha * num_pickers`.
- `--max-active-agvs`: limite absoluto (si se pasa, sobreescribe la regla).
- `--csv` / `--no-csv`: salida de resultados.

## 4) Diagrama Mermaid (alto nivel)

```mermaid
flowchart TB
  subgraph S[scripts/]
    E[eval.py CLI\n--env-id --policy --episodes --steps --seed\n--distance --active-alpha --max-active-agvs --csv]
  end

  subgraph X[tarware_ext/]
    R[Runner / rollout.py\nrollout(env, policy)]
    A[TarwareAdapter\nreset()/step() -> Transition]
    M[Metrics / metrics.py\nupdate() + finalize()]
    P1[HeuristicPolicy\n(episodic)]
    P2[GraphGreedyPolicy\n(step-wise)\ndistance_mode + active_alpha]
    T[Transition (normalized)\nobs\nreward_by_agent, reward_team\ndone_by_agent, done_all\ninfo]
  end

  subgraph C[tarware/ (core simulator)]
    ENV[Gym Env\nwarehouse.py + spaces/*\nreset()/step()]
    H[heuristic.py\n(baseline logic)]
  end

  E -->|gym.make(env_id)| ENV
  E -->|select policy| P1
  E -->|select policy| P2
  E -->|run episodes| R

  R --> A
  A -->|calls| ENV
  A -->|returns| T
  R -->|updates| M
  M -->|writes| CSV[(CSV file)]
  M -->|prints| OUT[Console summary]

  R -->|episodic path| P1
  P1 -->|delegates episode control| H

  R -->|step-wise path| P2
  P2 -->|act(env_unwrapped) -> actions| A

  E -.->|--distance manhattan/find_path| P2
  E -.->|--active-alpha / --max-active-agvs| P2
```

## 5) Ejemplo rapido

```bash
python scripts/eval.py \
  --env-id tarware-large-12agvs-7pickers-globalobs-v1 \
  --policy graph_greedy \
  --distance find_path \
  --active-alpha 3 \
  --episodes 5 \
  --steps 200 \
  --csv eval_graph_greedy_large_find.csv
```
