from train import train_dataset, inference, model_comparison
from absl import app, flags, logging
import os
import torch


FLAGS = flags.FLAGS
# parameters
flags.DEFINE_string("gpus", "3", "Ids of GPUs where the program run.")
flags.DEFINE_boolean("homogeneous", True, "whether to generate data with homogeneous intensity")
flags.DEFINE_float("intensity", 0.01, "spatial intensity")
flags.DEFINE_boolean("spline_param", True, "whether to use spline parameterised or gaussian kernel desgin matrix.")
flags.DEFINE_integer("spacing", 5, "spline spacing in x/y/z direction.")
flags.DEFINE_string("model", None, "whether to use Poisson/Negative model")
flags.DEFINE_string("penalty", None, "whether to use no penalty/firth-type penalty")
flags.DEFINE_integer("n_experiment", 100, "number of experiment")
flags.DEFINE_boolean("covariates", False, "whether consider study-level covariates into framework.")
flags.DEFINE_float("lr", 1, "learning rate.")


def main(argv):
    del argv  # Unused.  
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        device = 'cuda'
    else:
        device = 'cpu'
    logging.info("Device: {}".format(device))
    torch.cuda.empty_cache()
    train = train_dataset(homogeneous=FLAGS.homogeneous, intensity=FLAGS.intensity, covariates=FLAGS.covariates, spline_param=FLAGS.spline_param, spacing=FLAGS.spacing, device=device)
    train_model = train.train(model=FLAGS.model, penalty=FLAGS.penalty, n_experiment=FLAGS.n_experiment, iter=10000, lr=FLAGS.lr, tol=1e-9)
    # result_inference = inference(dataset=FLAGS.dataset, spacing=FLAGS.spacing, penalty=FLAGS.penalty, covariates=FLAGS.covariates)
    # predicted_intensity_image = result_inference.pred_image_all_models(mask='brain')
    # l = result_inference.model_selection_criteria()
    exit()
    
    
    models = model_comparison(spacing=FLAGS.spacing, penalty=FLAGS.penalty, covariates=FLAGS.covariates, device=device)
    likelihood_plot = models.likelihood_comparison()
    exit()
    avg_foci = models.foci_per_study()
    foci_sum = models.FociSum_boxplot()
    cov_mat = models.Cov_structure()
    Std_boxplot = models.Std_boxplot()
    var_compare = models.variance_comparison()
    var_bias_plot = models.var_boxplot()
    print('done')
    # K_function = models.K_function(h=10)
    # K_val_plot = models.K_val_plot()
    exit()
if __name__ == "__main__":
    app.run(main)

