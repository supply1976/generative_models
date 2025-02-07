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

def load_image_data(npzfile, raw01=False):
    images = np.load(npzfile)['images']
    if raw01:
      # (0, 1) -> (-1, 1)
      images = 2*images -1
    else:
      images = np.clip(images, -1, 1)
    n, h, w, c = images.shape
    if c < 3:
      images = np.concatenate([images, np.zeros([n, h, w, 3-c])], axis=-1)
    print(images.min(), images.max())
    return images

def kid_polynominal_kernel(features_1, features_2):
    batch, n = features_1.shape
    return (features_1 @ np.transpose(features_2) / n + 1.0)**3

def calculate_fid_kid(real_features, fake_features):
    n_real, fdim_real = real_features.shape
    n_fake, fdim_fake = fake_features.shape
    print(real_features.shape, fake_features.shape)
    
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
    if n_real != n_fake:
      n = np.min([n_real, n_fake])
    else:
      n = n_real
    kernel_real = kid_polynominal_kernel(real_features[0:n], real_features[0:n])
    kernel_fake = kid_polynominal_kernel(fake_features[0:n], fake_features[0:n])
    kernel_cros = kid_polynominal_kernel(real_features[0:n], fake_features[0:n])
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


def ddpm32_tr1000():
    # 10,000 epochs 1000 steps
    npz_gen6 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_1000steps_20250122-220425/cont_tr_linear_1000_20250128-223122/imgen_1000steps_20250130-080623/gen_1000x256x256x1_raw.npz"
    # 10,000 epochs 100 steps
    #npz_gen7 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_1000steps_20250122-220425/cont_tr_linear_1000_20250128-223122/imgen_100steps_20250207-031718/gen_1000x256x256x1_raw.npz"
    npz_gen7 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_1000steps_20250122-220425/cont_tr_linear_1000_20250128-223122/imgen_100steps_20250130-165951/gen_1000x256x256x1_raw.npz"
    # 10,000 epochs 100 steps, clip_denoise True
    npz_gen7b = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_1000steps_20250122-220425/cont_tr_linear_1000_20250128-223122/imgen_clipDN_100steps_20250206-205241/gen_1000x256x256x1_raw.npz"
    # 10,000 epochs 50 steps
    npz_gen8 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_1000steps_20250122-220425/cont_tr_linear_1000_20250128-223122/imgen_50steps_20250130-075350/gen_1000x256x256x1_raw.npz"
    # 10,000 epochs 20 steps
    npz_gen9 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_1000steps_20250122-220425/cont_tr_linear_1000_20250128-223122/imgen_20steps_20250130-074252/gen_1000x256x256x1_raw.npz"
    npz_gen9b = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_1000steps_20250122-220425/cont_tr_linear_1000_20250128-223122/imgen_20steps_20250207-035642/gen_1000x256x256x1_raw.npz"

    # 5,000 epochs 1000 steps 
    npz_gen1 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_1000steps_20250122-220425/imgen_20250124-001658/gen_1000x256x256x1_raw.npz"
    # 5,000 epochs 100 steps
    npz_gen2 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_1000steps_20250122-220425/imgen_20250124-070138/gen_1000x256x256x1_raw.npz"
    # 5,000 epochs 50 steps
    npz_gen3 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_1000steps_20250122-220425/imgen_50steps_20250128-220351/gen_1000x256x256x1_raw.npz"
    # 5,000 epochs 20 steps
    npz_gen4 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_1000steps_20250122-220425/imgen_20250124-072930/gen_1000x256x256x1_raw.npz"

    npz_real = "/remote/ltg_proj02_us01/user/richwu/datasets_for_ML_prototypes/metal_test1/pitch_8_512x512x1/all_images_713x256x256x1.npz"

    model = load_inception_model(MODEL_FILE, input_shape=(256,256,3))

    real_images = load_image_data(npz_real, raw01=True)
    gen1_images = load_image_data(npz_gen1, raw01=False)
    gen2_images = load_image_data(npz_gen2, raw01=False)
    gen3_images = load_image_data(npz_gen3, raw01=False)
    gen4_images = load_image_data(npz_gen4, raw01=False)
    gen6_images = load_image_data(npz_gen6, raw01=False)
    gen7_images = load_image_data(npz_gen7, raw01=False)
    gen7b_images = load_image_data(npz_gen7b, raw01=False)
    gen8_images = load_image_data(npz_gen8, raw01=False)
    gen9_images = load_image_data(npz_gen9, raw01=False)
    gen9b_images = load_image_data(npz_gen9b, raw01=False)
    
    real_features = model.predict(real_images, batch_size=32)
    gen1_features = model.predict(gen1_images, batch_size=32)
    gen2_features = model.predict(gen2_images, batch_size=32)
    gen3_features = model.predict(gen3_images, batch_size=32)
    gen4_features = model.predict(gen4_images, batch_size=32)
    gen6_features = model.predict(gen6_images, batch_size=32)
    gen7_features = model.predict(gen7_images, batch_size=32)
    gen7b_features = model.predict(gen7b_images, batch_size=32)
    gen8_features = model.predict(gen8_images, batch_size=32)
    gen9_features = model.predict(gen9_images, batch_size=32)
    gen9b_features = model.predict(gen9b_images, batch_size=32)

    fid1, kid1 = calculate_fid_kid(real_features, gen1_features)
    fid2, kid2 = calculate_fid_kid(real_features, gen2_features)
    fid3, kid3 = calculate_fid_kid(real_features, gen3_features)
    fid4, kid4 = calculate_fid_kid(real_features, gen4_features)
    fid6, kid6 = calculate_fid_kid(real_features, gen6_features)
    fid7, kid7 = calculate_fid_kid(real_features, gen7_features)
    fid7b, kid7b = calculate_fid_kid(real_features, gen7b_features)
    fid8, kid8 = calculate_fid_kid(real_features, gen8_features)
    fid9, kid9 = calculate_fid_kid(real_features, gen9_features)
    fid9b, kid9b = calculate_fid_kid(real_features, gen9b_features)
    
    print("FID and KID: real vs gen1 ", fid1, np.sqrt(fid1), kid1)
    print("FID and KID: real vs gen2 ", fid2, np.sqrt(fid2), kid2)
    print("FID and KID: real vs gen3 ", fid3, np.sqrt(fid3), kid3)
    print("FID and KID: real vs gen4 ", fid4, np.sqrt(fid4), kid4)
    print("FID and KID: real vs gen6 ", fid6, np.sqrt(fid6), kid6)
    print("FID and KID: real vs gen7 ", fid7, np.sqrt(fid7), kid7)
    print("FID and KID: real vs gen7b ", fid7b, np.sqrt(fid7b), kid7b)
    print("FID and KID: real vs gen8 ", fid8, np.sqrt(fid8), kid8)
    print("FID and KID: real vs gen9 ", fid9, np.sqrt(fid9), kid9)
    print("FID and KID: real vs gen9b ", fid9b, np.sqrt(fid9b), kid9b)

def ddpm64_tr1000():
    # 10,000 epochs 1000 steps
    # 10,000 epochs 100 steps
    # 10,000 epochs 50 steps
    # 10,000 epochs 20 steps

    # 5,000 epochs 1000 steps 
    npz_gen1 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM64_256x256x1__trset_metal_test1/linear_1000_20250128-223817/imgen_1000steps_20250205-070337/gen_1000x256x256x1_raw.npz"
    # 5,000 epochs 100 steps
    npz_gen2 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM64_256x256x1__trset_metal_test1/linear_1000_20250128-223817/imgen_100steps_20250205-052620/gen_1000x256x256x1_raw.npz"
    # 5,000 epochs 50 steps
    npz_gen3 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM64_256x256x1__trset_metal_test1/linear_1000_20250128-223817/imgen_50steps_20250205-063324/gen_1000x256x256x1_raw.npz"
    # 5,000 epochs 20 steps
    npz_gen4 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM64_256x256x1__trset_metal_test1/linear_1000_20250128-223817/imgen_20steps_20250205-065359/gen_1000x256x256x1_raw.npz"

    npz_real = "/remote/ltg_proj02_us01/user/richwu/datasets_for_ML_prototypes/metal_test1/pitch_8_512x512x1/all_images_713x256x256x1.npz"

    model = load_inception_model(MODEL_FILE, input_shape=(256,256,3))

    real_images = load_image_data(npz_real, raw01=True)
    gen1_images = load_image_data(npz_gen1, raw01=False)
    gen2_images = load_image_data(npz_gen2, raw01=False)
    gen3_images = load_image_data(npz_gen3, raw01=False)
    gen4_images = load_image_data(npz_gen4, raw01=False)
    #gen6_images = load_image_data(npz_gen6, raw01=False)
    #gen7_images = load_image_data(npz_gen7, raw01=False)
    #gen8_images = load_image_data(npz_gen8, raw01=False)
    #gen9_images = load_image_data(npz_gen9, raw01=False)
    
    real_features = model.predict(real_images, batch_size=32)
    gen1_features = model.predict(gen1_images, batch_size=32)
    gen2_features = model.predict(gen2_images, batch_size=32)
    gen3_features = model.predict(gen3_images, batch_size=32)
    gen4_features = model.predict(gen4_images, batch_size=32)
    #gen6_features = model.predict(gen6_images, batch_size=32)
    #gen7_features = model.predict(gen7_images, batch_size=32)
    #gen8_features = model.predict(gen8_images, batch_size=32)
    #gen9_features = model.predict(gen9_images, batch_size=32)

    fid1, kid1 = calculate_fid_kid(real_features, gen1_features)
    fid2, kid2 = calculate_fid_kid(real_features, gen2_features)
    fid3, kid3 = calculate_fid_kid(real_features, gen3_features)
    fid4, kid4 = calculate_fid_kid(real_features, gen4_features)
    #fid6, kid6 = calculate_fid_kid(real_features, gen6_features)
    #fid7, kid7 = calculate_fid_kid(real_features, gen7_features)
    #fid8, kid8 = calculate_fid_kid(real_features, gen8_features)
    #fid9, kid9 = calculate_fid_kid(real_features, gen9_features)
    
    print("FID and KID: real vs gen1 ", fid1, np.sqrt(fid1), kid1)
    print("FID and KID: real vs gen2 ", fid2, np.sqrt(fid2), kid2)
    print("FID and KID: real vs gen3 ", fid3, np.sqrt(fid3), kid3)
    print("FID and KID: real vs gen4 ", fid4, np.sqrt(fid4), kid4)
    #print("FID and KID: real vs gen6 ", fid6, np.sqrt(fid6), kid6)
    #print("FID and KID: real vs gen7 ", fid7, np.sqrt(fid7), kid7)
    #print("FID and KID: real vs gen8 ", fid8, np.sqrt(fid8), kid8)
    #print("FID and KID: real vs gen9 ", fid9, np.sqrt(fid9), kid9)

def ddpm32_tr100():
    # 10,000 epochs 100 steps
    npz_gen4 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_100steps_20250122-220803/cont_tr_linear_100_20250126-223747/imgen_20250128-180353/gen_1000x256x256x1_raw.npz"
    # 10,000 epochs 50 steps
    npz_gen5 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_100steps_20250122-220803/cont_tr_linear_100_20250126-223747/imgen_50steps_20250202-153741/gen_1000x256x256x1_raw.npz"
    # 10,000 epochs 20 steps
    npz_gen6 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_100steps_20250122-220803/cont_tr_linear_100_20250126-223747/imgen_20250128-183344/gen_1000x256x256x1_raw.npz"

    # 5,000 epochs 100 steps 
    npz_gen1 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_100steps_20250122-220803/imgen_20250123-232832/gen_1000x256x256x1_raw.npz"
    # 5,000 epochs 50 steps
    npz_gen2 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_100steps_20250122-220803/imgen_50steps_20250202-152441/gen_1000x256x256x1_raw.npz"
    # 5,000 epochs 20 steps
    npz_gen3 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/linear_100steps_20250122-220803/imgen_20250124-064947/gen_1000x256x256x1_raw.npz"

    npz_real = "/remote/ltg_proj02_us01/user/richwu/datasets_for_ML_prototypes/metal_test1/pitch_8_512x512x1/all_images_713x256x256x1.npz"

    model = load_inception_model(MODEL_FILE, input_shape=(256,256,3))

    real_images = load_image_data(npz_real, raw01=True)
    gen1_images = load_image_data(npz_gen1, raw01=False)
    gen2_images = load_image_data(npz_gen2, raw01=False)
    gen3_images = load_image_data(npz_gen3, raw01=False)
    gen4_images = load_image_data(npz_gen4, raw01=False)
    gen5_images = load_image_data(npz_gen5, raw01=False)
    gen6_images = load_image_data(npz_gen6, raw01=False)
    
    real_features = model.predict(real_images, batch_size=32)
    gen1_features = model.predict(gen1_images, batch_size=32)
    gen2_features = model.predict(gen2_images, batch_size=32)
    gen3_features = model.predict(gen3_images, batch_size=32)
    gen4_features = model.predict(gen4_images, batch_size=32)
    gen5_features = model.predict(gen5_images, batch_size=32)
    gen6_features = model.predict(gen6_images, batch_size=32)

    fid1, kid1 = calculate_fid_kid(real_features, gen1_features)
    fid2, kid2 = calculate_fid_kid(real_features, gen2_features)
    fid3, kid3 = calculate_fid_kid(real_features, gen3_features)
    fid4, kid4 = calculate_fid_kid(real_features, gen4_features)
    fid5, kid5 = calculate_fid_kid(real_features, gen5_features)
    fid6, kid6 = calculate_fid_kid(real_features, gen6_features)
    
    print("FID and KID: real vs gen1 ", fid1, np.sqrt(fid1), kid1)
    print("FID and KID: real vs gen2 ", fid2, np.sqrt(fid2), kid2)
    print("FID and KID: real vs gen3 ", fid3, np.sqrt(fid3), kid3)
    print("FID and KID: real vs gen4 ", fid4, np.sqrt(fid4), kid4)
    print("FID and KID: real vs gen5 ", fid4, np.sqrt(fid5), kid5)
    print("FID and KID: real vs gen6 ", fid4, np.sqrt(fid6), kid6)


def ddpm32_tr1000_cosine():
    # 10,000 epochs 100 steps
    # 10,000 epochs 50 steps
    # 10,000 epochs 20 steps

    # 5,000 epochs 100 steps 
    # 5,000 epochs 50 steps
    # 5,000 epochs 20 steps
    npz_gen3 = "/remote/ltg_proj02_us01/user/richwu/GIT_PROJECTS/diffpattern/py_scripts/ddpm_v2/training_outputs/DDPM32_256x256x1__trset_metal_test1/cosine_1000_20250205-184410/imgen_20steps_20250207-035356/gen_1000x256x256x1_raw.npz"

    npz_real = "/remote/ltg_proj02_us01/user/richwu/datasets_for_ML_prototypes/metal_test1/pitch_8_512x512x1/all_images_713x256x256x1.npz"

    model = load_inception_model(MODEL_FILE, input_shape=(256,256,3))

    real_images = load_image_data(npz_real, raw01=True)
    #gen1_images = load_image_data(npz_gen1, raw01=False)
    #gen2_images = load_image_data(npz_gen2, raw01=False)
    gen3_images = load_image_data(npz_gen3, raw01=False)
    
    real_features = model.predict(real_images, batch_size=32)
    gen3_features = model.predict(gen3_images, batch_size=32)

    fid3, kid3 = calculate_fid_kid(real_features, gen3_features)
    
    #print("FID and KID: real vs gen1 ", fid1, np.sqrt(fid1), kid1)
    #print("FID and KID: real vs gen2 ", fid2, np.sqrt(fid2), kid2)
    print("FID and KID: real vs gen3 ", fid3, np.sqrt(fid3), kid3)
    #print("FID and KID: real vs gen4 ", fid4, np.sqrt(fid4), kid4)
    #print("FID and KID: real vs gen5 ", fid4, np.sqrt(fid5), kid5)
    #print("FID and KID: real vs gen6 ", fid4, np.sqrt(fid6), kid6)

if __name__=="__main__":
    #unit_test()
    #ddpm64_tr1000()
    ddpm32_tr1000_cosine()


