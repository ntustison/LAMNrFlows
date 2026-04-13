#!/user/bin/bash

iterations=100
aug_params_dlbs="noise_std:cos:0.05->0.015@${iterations},\
sd_affine:cos:0.05->0.01@${iterations},\
sd_deformation:linear:12.0->0.6@${iterations},\
sd_simulated_bias_field:cos:0.20->0.03@${iterations},\
sd_histogram_warping:cos:0.04->0.008@${iterations}"

WHICH_PYTHON="/Users/ntustison/anaconda3/bin/python3"
${WHICH_PYTHON} -m pytest -q -s ~/Pkg/ANTsTorch/tests/test_image_dataset_and_scheduler.py -vv \
  --aug-schedules ${aug_params_dlbs} \
  --aug-steps ${iterations} --dump-aug-samples --grid 10 --tile-size 128 --preview-channel 0 --mods T1

iterations=100
aug_params_ppmi="noise_std:cos:0.05->0.004@${iterations},\
sd_affine:cos:0.05->0.00@${iterations},\
sd_deformation:linear:12.0->0.6@${iterations},\
sd_simulated_bias_field:cos:0.20->0.03@${iterations},\
sd_histogram_warping:cos:0.04->0.008@${iterations}"

WHICH_PYTHON="/Users/ntustison/anaconda3/bin/python3"
${WHICH_PYTHON} -m pytest -q -s ~/Pkg/ANTsTorch/tests/test_image_dataset_and_scheduler.py -vv \
  --aug-schedules ${aug_params_ppmi} \
  --aug-steps ${iterations} --dump-aug-samples --grid 10 --tile-size 128 --preview-channel 0 --mods T1

# ffmpeg -framerate 5 -pattern_type glob -i 'aug_step*.png' -c:v libx264 -pix_fmt yuv420p aug_ldbs.mp4