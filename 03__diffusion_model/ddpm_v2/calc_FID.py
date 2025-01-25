import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy.linalg import sqrtm


MODEL_FILE = "/remote/us01home28/richwu/Downloads/inception_v3_weights_tf_dim_ordering_tf_kernels.h5"

def load_inception_model(model_path, input_shape=(256,256,3)):
    orig_model = tf.keras.applications.InceptionV3(
        include_top=True, 
        weights=model_path,
        input_shape=input_shape,
        pooling='avg')
    # remove last FC layer, for FID, KID usage
    model = keras.Model(
        inputs=orig_model.layers[1].input, 
        outputs=orig_model.layers[-2].output)
    return model


def kid_polynominal_kernel(features_1, features_2):
    batch, n = features_1.shape
    return (features_1 @ np.transpose(features_2) / n + 1.0)**3

def calculate_fid_kid(inception_model, real_images, fake_images, batch_size=32):
    n, _, _, _ = real_images.shape
    # Calculate features for real and fake images
    real_features = inception_model.predict(real_images, batch_size=batch_size)
    fake_features = inception_model.predict(fake_images, batch_size=batch_size)
    
    # Calculate mean and covariance of real and generated activations
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Calculate FID score
    ssdiff = np.sum((mu_real - mu_fake)**2)
    covmean = sqrtm(sigma_real.dot(sigma_fake))
    if np.iscomplexobj(covmean):
      covmean = covmean.real
    fid = ssdiff + np.trace(sigma_real + sigma_fake - 2 * covmean)

    # calculate KID
    kernel_real = kid_polynominal_kernel(real_features, real_features)
    kernel_fake = kid_polynominal_kernel(fake_features, fake_features)
    kernel_cros = kid_polynominal_kernel(real_features, fake_features)
    kid_mean_real = np.sum(kernel_real * (1-np.eye(n))) / (n*(n-1.0))
    kid_mean_fake = np.sum(kernel_fake * (1-np.eye(n))) / (n*(n-1.0))
    kid_mean_cros = np.mean(kernel_cros)
    kid = kid_mean_real + kid_mean_fake -2.0 *kid_mean_cros
    return (fid, kid)


def unit_test():
    n_imgs = 713
    model = load_inception_model(MODEL_FILE, input_shape=(256,256,3))
    #np.random.seed(1)
    real_images = np.random.randn(n_imgs,256,256,3)
    fake_images = np.random.randn(n_imgs,256,256,3)
    real_images = np.clip(real_images, -1, 1)
    fake_images = np.clip(fake_images, -1, 1)

    fid, kid = calculate_fid_kid(model, real_images, fake_images)
    print(fid, kid)


def main():
    npz_gen2 = "./training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_1000steps_20250122-220425/cont_tr_linear_1000_20250124-080900/imgen_20250124-154518/gen_1000x256x256x1_raw.npz"
    #"/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_1000steps_20250122-220425/imgen_20250124-072930/gen_1000x256x256x1_raw.npz"
    #"/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_1000steps_20250122-220425/imgen_20250124-070138/gen_1000x256x256x1_raw.npz"
    #"/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_100steps_20250122-220803/imgen_20250124-064947/gen_1000x256x256x1_raw.npz"
    npz_gen1 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_1000steps_20250122-220425/imgen_20250124-001658/gen_1000x256x256x1_raw.npz"
    #npz_gen1 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_100steps_20250122-220803/imgen_20250123-232832/gen_1000x256x256x1_raw.npz"
    #npz_gen1 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_20steps_20250122-221154/imgen_20250124-000723/gen_1000x256x256x1_raw.npz"
    npz_real = "/remote/ltg_proj02_us01/user/richwu/datasets_for_ML_prototypes/metal_test1/pitch_8_512x512x1/all_images_713x256x256x1.npz"

    model = load_inception_model(MODEL_FILE, input_shape=(256,256,3))

    real_images = np.load(npz_real)['images']
    real_images = 2*real_images -1
    n, h, w, c = real_images.shape
    real_images = np.concatenate([real_images, np.zeros([n, h, w, 3-c])], axis=-1)
    print(real_images.max(), real_images.min())

    gen1_images = np.load(npz_gen1)['images']
    gen1_images = np.clip(gen1_images[range(n)], -1, 1)
    #gen1_images = 0.5*(gen1_images+1)
    gen1_images = np.concatenate([gen1_images, np.zeros([n, h, w, 3-c])], axis=-1)
    print(gen1_images.max(), gen1_images.min())
    
    gen2_images = np.load(npz_gen2)['images']
    gen2_images = np.clip(gen2_images[range(n)], -1, 1)
    #gen2_images = 0.5*(gen2_images+1)
    gen2_images = np.concatenate([gen2_images, np.zeros([n, h, w, 3-c])], axis=-1)
    print(gen2_images.max(), gen2_images.min())

    #noise_images = np.random.randint(0,2, size=(n, h, w, 3)).astype(np.float64)
    #noise_images = 2*noise_images -1
    noise_images = np.random.randn(n, h, w, 3)
    noise_images = np.clip(noise_images, -1, 1)

    fid1, kid1 = calculate_fid_kid(model, real_images, gen1_images)
    fid2, kid2 = calculate_fid_kid(model, real_images, gen2_images)
    fid3, kid3 = calculate_fid_kid(model, gen1_images, gen2_images)
    print("FID and KID: real vs gen1 ", fid1, kid1)
    print("FID and KID: real vs gen2 ", fid2, kid2)
    print("FID and KID: gen1 vs gen2 ", fid3, kid3)

if __name__=="__main__":
    #unit_test()
    main()


