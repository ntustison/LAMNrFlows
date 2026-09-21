total_iter=5

aug_params="noise_std:cos:0.06->0.008@${total_iter},\
sd_affine:cos:0.05->0.00@${total_iter},\
sd_deformation:linear:16.0->1.0@${total_iter},\
sd_simulated_bias_field:cos:0.25->0.05@${total_iter},\
sd_histogram_warping:cos:0.05->0.01@${total_iter}"

pytest -q -s ~/Pkg/ANTsTorch/tests/test_image_dataset_and_scheduler.py -vv \
  --aug-schedules ${aug_params} \
  --aug-steps ${total_iter} --dump-aug-samples --grid 5 --tile-size 128 --preview-channel 0