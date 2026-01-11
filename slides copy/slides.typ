#import "@preview/polylux:0.4.0": *
#import "@preview/metropolis-polylux:0.1.0" as metropolis
#import metropolis: focus, new-section

#show: metropolis.setup

#slide[
  #set page(header: none, footer: none, margin: 3em)


  #text(size: 1.3em)[
    *PHM Society 2025 Data Challenge*
  ]

  DIT Semestral Project

  #metropolis.divider

  #set text(size: .8em, weight: "light")
  Ondřej Baštař

  Jan 4, 2025

  Some extra info
]

#slide[
  = Agenda

  #metropolis.outline
]

#new-section[Problem Description]


#slide[
  = Problem statement

  #figure(caption: "Problem diagram")[

    #image("problem_diagram.png")
  ]
]

#slide()[
  = Dataset Specifications

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    align: top,

    // Left Column: Inputs
    [
      #set block(fill: rgb("#f0f4f8"), inset: 8pt, radius: 4pt)
      // #block[
      //   *Metadata (Context)*
      //   - *ESN:* Engine Serial Number
      //   - *Age:* Cycles Since New
      //   - *History:* Cumulative counts for Shop Visits (HPC, HPT) & Water Washes
      // ]

      #block[
        *Sensor Data (16 Features)*
        - *Flight:* $"Alt", "Ma", P_("amb")$
        - *Temps:* $T_(t_2), T_(2.5), T_3, T_(4.5), T_5$
        - *Pressures:* $P_(t_2), P_(s_3), P_(2.5)$
        - *Rotors:* $N_1$ (Fan), $N_2$ (Core)
        - *Actuators:* $"VAFN", "VBV"$
      ]
    ],

    // Right Column: Outputs
    [
      #set block(fill: rgb("#e8f5e9"), inset: 8pt, radius: 4pt)
      #block[
        *Prediction Targets (RUL)*
        1. *HPT Shop Visit* \
          (High Pressure Turbine)
        2. *HPC Shop Visit* \
          (High Pressure Compressor)
        3. *Water Wash* \
          (Routine Maintenance)
      ]

    ],
  )
]

#slide[
  = Dataset Specifications
  Rough amounts of data:
  - Mesaurement from 4 engines
  - \~2000 datapoints per engine at 8 flight states (landed, flying, takeoff, ...)
  - Events measured per engine:
    - WW (Water Wash): \~20 per engine
    - HPC Shop Visit: \~3 per engine
    - HPT Shop Visit: \~6 per engine

  #set block(fill: rgb("#f0f4f8"), inset: 8pt, radius: 4pt)
  #block()[ In conclusion the data are limited for advanced machine learning methods. ]
]

#slide[
  = Model Evaluation
  - Competition scoring function
  $
     W(y,hat(y)) & = cases(
                     2/(1 + 0.02 dot.c y) "if" hat(y) >= y " (Late prediction)",
                     1/(1 + 0.02 dot.c y) "if" hat(y) < y " (Early prediction)",
                   ) \
    S(y, hat(y)) & = W(y, hat(y)) dot.c (hat(y) - y)^2
  $

]

#new-section[My Approach]

#slide[
  = Baseline Model
  #grid(columns: (1fr, 1fr))[
    - Simple baseline model for comparison.
    - Based on XGBoost (Gradient Boosting Trees)
    - Mean score on 4 engines is *93.60*
  ][
    #figure()[
      #table(
        columns: (auto, auto, auto, auto, auto),
        inset: 10pt,
        align: center,
        // Add fills for header and the Mean row
        fill: (col, row) => {
          if row == 0 { luma(230) } // Light gray header
          else if row == 6 { rgb("e6f7ff") } // Highlight Mean row (Light Blue)
          else { white }
        },

        table.header([*Engine SN*], [*WW*], [*HPC*], [*HPT*], [*Final*]),

        [101], [38.17], [84.66], [136.81], [86.55],
        [102], [50.56], [203.16], [78.84], [110.85],
        [103], [42.66], [159.65], [42.89], [81.73],
        [104], [53.60], [167.37], [64.82], [95.26],
        [], [], [], [], [],
        [*Mean*], [46.25], [153.71], [80.84], [*93.60*],
        // Bold final result
        [*Std*], [7.09], [49.78], [40.14], [12.79],
      )
    ]
  ]

]

#slide[
  = Kalman Filter for health tracking
  1. Trained a simple model for healthy engine operation such that:
    - $f("flight state") = (hat(T)_1, hat(T)_2, ... )^T = hat(bold(z))_k$
  2. Calculate residuals:
    - $bold(r)_k = bold(z)_k - hat(bold(z))_k$
    - $bold(z)_k$ is vector of sensor values at time $k$.
    - size of elements of vector $bold(r)_k$ correlates with engine health.
  3. Kalman filter for tracking health index for each sesnor residual.
    - Using N simple constant velocity models with each element of $bold(r)_k$ as outputs.
  $ bold(x)_k = vec(x_1, x_2) = vec("Health (residual)", "Degradation rate") quad A = mat(1, Delta t; 0, 1) $

]

#slide[
  #show: focus
  Something very important
]
