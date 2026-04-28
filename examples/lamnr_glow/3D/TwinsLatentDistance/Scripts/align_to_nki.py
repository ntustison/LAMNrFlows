import ants
import antspynet
import os
import glob

base_dir = "/Users/ntustison/Data/Public/OpenNeuro/ds004169/"
nki_template_file = base_dir + "Template/nki_x_brain.nii.gz"
nki_template = ants.image_read(nki_template_file)

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "4"

t1_files = glob.glob(os.path.join(base_dir, "OriginalDownload/sub-*/ses-01/anat/sub-*_T1w.nii.gz"))

for t1_file in t1_files:
    print(f"Processing {t1_file}")

    output_dir = os.path.dirname(t1_file).replace("OriginalDownload", "BIDSAlignedToTemplate")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(t1_file))
    output_brain_file = os.path.join(output_dir, os.path.basename(t1_file).replace("_T1w.nii.gz", "_T1wbrain.nii.gz"))
    output_prefix = os.path.join(output_dir, os.path.basename(t1_file).replace("_T1w.nii.gz", "_TemplatexT1w_"))
    
    t2_files = glob.glob(os.path.dirname(t1_file) + "/*T2w.nii.gz")

    t2_file=''
    output_t2_file=""
    if len(t2_files) > 0:
        t2_file = t2_files[0]
        output_t2_file = os.path.join(output_dir, os.path.basename(t2_file))
        
    if os.path.exists(output_t2_file):
        continue

    if not os.path.exists(output_file) or not os.path.exists(output_brain_file):
        t1_image = ants.image_read(t1_file)

        t1_mask_3 = antspynet.brain_extraction(t1_image, modality="t1threetissue", verbose=False)['segmentation_image']
        t1_mask = ants.threshold_image(t1_mask_3, 1, 1, 1, 0)

        t1_n4 = ants.n4_bias_field_correction(t1_image, verbose=True)
        t1_n4_brain = t1_n4 * t1_mask
        reg = ants.registration(fixed=nki_template, moving=t1_n4_brain, outprefix=output_prefix, type_of_transform="antsRegistrationSyNQuick[r]", verbose=True)
        aligned_t1 = ants.apply_transforms(nki_template, t1_n4, reg["fwdtransforms"], interpolator="linear", verbose=True)
        ants.image_write(aligned_t1, output_file)
        ants.image_write(reg['warpedmovout'], output_brain_file)

        if os.path.exists(t2_file):
            t2_image = ants.image_read(t2_file)

            t2_mask_3 = antspynet.brain_extraction(t2_image, modality="flair", verbose=False)
            t2_mask = ants.threshold_image(t2_mask_3, 0.5, 1, 1, 0)
            
            t2_n4 = ants.n4_bias_field_correction(t2_image, verbose=True)
            t2_n4_brain = t2_n4 * t2_mask
            reg_t1xt2 = ants.registration(fixed=t1_n4_brain, moving=t2_n4_brain, type_of_transform="antsRegistrationSyNQuick[r]", verbose=True)
            aligned_t2 = ants.apply_transforms(nki_template, t2_n4, list((*reg["fwdtransforms"], *reg_t1xt2["fwdtransforms"])), interpolator="linear", verbose=True)
            ants.image_write(aligned_t2, output_t2_file)



