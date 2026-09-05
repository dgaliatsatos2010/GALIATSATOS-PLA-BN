import numpy as np

from galiatsatos_plabn import (
    GaliatsatosMethod,
    make_ordered_threshold_operator,
)

C1 = make_ordered_threshold_operator(3, threshold=1)
C2 = make_ordered_threshold_operator(3, threshold=2)

X1 = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [2, 1],
    [2, 2],
])

y1 = np.array([0, 1, 1, 1, 1, 1])

X2 = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [2, 1],
    [2, 2],
    [2, 1],
])

y2 = np.array([0, 0, 0, 1, 1, 1])

model = GaliatsatosMethod(
    n_classes=3,
    structure="tan",
    smoothing=0.10,
    max_iter=120,
    tol=1e-5,
    n_init=3,
    init_jitter=0.05,
    random_state=0,
).fit(
    [X1, X2],
    [y1, y2],
    [C1, C2],
)

print("Canonical posterior:")
print(model.transform(X1[:2]))

print("\nTransported probabilities:")
print(model.transport_proba(X1[:2], C2))

print("\nDiagnostics:")
print(model.method_diagnostics())
