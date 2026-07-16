----------------------------------------------
20260703

# 训练代码记录uppper bounding 相关值
# 训练代码要加载除去upper bounding 的其他值（网络参数自定义）
# 训练代码要加载除去upper bounding mean，并保持不变


CUDA_VISIBLE_DEVICES=0 python cifar10/train_count.py -c cifar10/cifar10.yml --model-file A_model_4layers_baseline_ILIF01_count_add_internal -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 
d

--------------------


CUDA_VISIBLE_DEVICES=1 python cifar10/train_count.py -c cifar10/cifar10.yml --model-file A_model_4layers_baseline_ILIF01_count_add_internal_kernel3 -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 
e






----------------------------------------------
20260704


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_add_internal -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9
d



CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_add_internal_kernel3 -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 

e



----------------------------------------------
20260705


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_fix_UPB_add_internal_kernel3 -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train/A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_add_internal_kernel3-20260704-142823/model_best.pth.tar
d



CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_fix_UPB_add_internal_kernel3_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train/A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_add_internal_kernel3-20260704-142823/model_best.pth.tar

e




--------------------------------------------
20260706


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_fix_UPB_add_internal_kernel3_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.99 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train/A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_add_internal_kernel3-20260704-142823/model_best.pth.tar

d


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_fix_UPB_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.99 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/output/train/model_4layers_baseline_ILIF01_count-20260627-003741/model_best.pth.tar

e




----------------------------------------------------------------

CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_fix_UPB_add_internal_kernel3_noatten -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.8 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train/A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_add_internal_kernel3-20260704-142823/model_best.pth.tar



CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_fix_UPB_add_internal_kernel3_noatten_test -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/A_output_test/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.7 --initial-checkpoint /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train/A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_add_internal_kernel3-20260704-142823/model_best.pth.tar







--------------------------------------------
20260706



CUDA_VISIBLE_DEVICES=1 python cifar10/train_count.py -c cifar10/cifar10.yml --model-file A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_add_internal_kernel3_count_mem_del_max -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 

a


CUDA_VISIBLE_DEVICES=1 python cifar10/train_count.py -c cifar10/cifar10.yml --model-file A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_add_internal_kernel3_count_mem_del_attention -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 

b















CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_add_internal -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 

e
















CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_online_add_internal -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.99 


--------------------


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_fix_UPB_add_internal -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.9 --initial-checkpoint 



CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive_only_load_upperbound.py -c cifar10/cifar10.yml --model-file A_model_4layers_baseline_ILIF01_count_to_adaptive_sj_back_fix_UPB_add_internal -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/A_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1 --spike-selected-percentile 0.99 --initial-checkpoint 