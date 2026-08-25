# 对每个数据用不同的参数
# target_methy, gwm, end_motif, flen, total, cna

script_path="/personal/nipgt_code_review/code_public/NICE/stage2_evaluation/nice.py"
data_path="/personal/nipgt_code_review/code_public/NICE/data/decent_low80"
save_path="/personal/nipgt_code_review/code_public/NICE/main_res"

# target_methy
python $script_path --data_path ${data_path}/target_methy.csv --save_path $save_path --k_features 10 --valid_ratio 1 --resample --n_trials 30 --thereshold 5 --n_c 0.95 --need_preprocess --n_jobs 12
# target_methy (save deployable final model)
# python $script_path --data_path ${data_path}/target_methy.csv --save_path $save_path --k_features 10 --valid_ratio 1 --resample --n_trials 30 --thereshold 5 --n_c 0.95 --need_preprocess --n_jobs 12 --save_final_model --final_model_name final_model.joblib
# # gwm
# python $script_path --data_path ${data_path}/gwm.csv --save_path $save_path --k_features 10 --valid_ratio 1 --n_trials 30 --thereshold 5 --n_c 0.95 --need_preprocess --n_jobs 1
# # end_motif
# python $script_path --data_path ${data_path}/end_motif.csv --save_path $save_path --k_features 30 --valid_ratio 1 --resample --n_trials 30 --thereshold 5 --n_c 0.95 --need_preprocess --n_jobs 1
# # flen
# python $script_path --data_path ${data_path}/flen.csv --save_path $save_path --k_features 3 --valid_ratio 1 --n_trials 30 --thereshold 5 --n_c 0.95 --need_preprocess --n_jobs 1
# # total
# python $script_path --data_path ${data_path}/total.csv --save_path $save_path --k_features 10 --valid_ratio 0.9 --resample --n_trials 30 --thereshold 5 --n_c 0.95 --need_preprocess --n_jobs 1
# # cna
# python $script_path --data_path ${data_path}/cna.csv --save_path $save_path --k_features 10 --valid_ratio 1 --resample --n_trials 30 --thereshold 5 --n_c 0.95 --need_preprocess --n_jobs 1
# # concat
# python $script_path --data_path ${data_path}/concat.csv --save_path $save_path --k_features 30 --valid_ratio 1 --resample --n_trials 30 --thereshold 5 --n_c 0.95 --n_jobs 1

