from train import train_dataset
from absl import app, flags, logging
import os
import torch


FLAGS = flags.FLAGS
# parameters
flags.DEFINE_string("gpus", "3", "Ids of GPUs where the program run.")
flags.DEFINE_integer("foci_per_study", 10, "Average number of foci per study")
flags.DEFINE_integer("spline_degree", 3, "the degress spline functions")
flags.DEFINE_boolean("half_interval_shift", False, "whether to shift the knots by a half interval")
flags.DEFINE_float("spacing", 5, "spline spacing in x/y/z direction.")
flags.DEFINE_string("model", None, "whether to use Poisson/Negative model")
flags.DEFINE_string("penalty", 'No', "whether to use no penalty/firth-type penalty")
flags.DEFINE_boolean("covariates", None, "whether consider study-level covariates into framework.")
flags.DEFINE_integer("n_experiment", 100, "the number of MC simulations")
flags.DEFINE_float("lr", 1, "learning rate.")
flags.DEFINE_float("tol", 1e-4, "the stopping criteria for convergence")


def main(argv):
    del argv  # Unused.  
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        device = 'cuda'
    else:
        device = 'cpu'
    logging.info("Device: {}".format(device))
    torch.cuda.empty_cache()
    train_model = train_dataset(foci_per_study=FLAGS.foci_per_study, model=FLAGS.model, 
                                covariates=FLAGS.covariates, penalty=FLAGS.penalty, spline_degree=FLAGS.spline_degree, 
                                half_interval_shift=FLAGS.half_interval_shift, spacing=FLAGS.spacing, device=device)
    train = train_model.outcome(n_experiment=FLAGS.n_experiment, lr=FLAGS.lr, tol=FLAGS.tol)
    
if __name__ == "__main__":
    app.run(main)

