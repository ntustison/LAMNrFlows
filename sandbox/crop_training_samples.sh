

base_dir="/Users/ntustison/Desktop/lamnr_glow_dlbs"
input_dir="${base_dir}/runs2d/dlbs_t1_t2flair_fa_96x128_K12_L5_HC256_Round2/"
output_dir="${base_dir}/TrainingSamplesCropped/"
mkdir -p "${output_dir}"

for i in `ls ${input_dir}/samples_*.png`; do
  echo ${i}
  img_base=$(basename ${i})
  img_output="${output_dir}/${img_base}"
  magick "${i}" -crop 512x480+0+0 +repage -rotate 90 "${img_output}"
done

ffmpeg -pattern_type glob -i "${output_dir}/samples_view0_*.png" \
       -pattern_type glob -i "${output_dir}/samples_view1_*.png" \
       -pattern_type glob -i "${output_dir}/samples_view2_*.png" \
       -filter_complex "[0:v][1:v][2:v]hstack=inputs=3,scale=500:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=32[p];[b][p]paletteuse=dither=bayer:bayer_scale=3" \
       -fps_mode passthrough \
       "${base_dir}/training_evolution_dlbs_2d_views.gif"