from tqdm import tqdm

from .convergence import convergence_monitor
from .utils import theta_mle


def matrix_completion(
    model,
    X,
    extra_metrics=None,
    fname="",
    epochs_per_val=5,
    num_epochs=2000,
    patience=200,
):
    """Run matrix completion on input matrix X using a factorization model.

    extra_metrics: iterable of name, exectuable pairs for extra metric logggin.
            iterable must have the signature (model: Type[BaseMF]) -> Float
    """

    # Results collected from the process
    output = {
        "convergence_rate": [],
        "loss_values": [],
        "epochs": [],
        "U": None,
        "V": None,
        "M": None,
        "s": None,
        "theta_mle": None,
    }

    for metric, _ in extra_metrics:
        output[metric] = []

    for epoch in tqdm(range(num_epochs)):

        model.run_step()

        output["epochs"].append(int(epoch))
        output["loss_values"].append(float(model.loss()))
        for metric, callable in extra_metrics:
            output[metric].append(callable(model))

        if epoch == patience:
            monitor = convergence_monitor(model.M)

        if epoch % epochs_per_val == 0 and epoch > patience:

            if monitor.converged(model.M):
                break

    output["U"] = model.U
    output["V"] = model.V
    output["M"] = model.M

    if hasattr(model, "s"):
        output["s"] = model.s

    output["theta_mle"] = theta_mle(X, model.M)

    return output
