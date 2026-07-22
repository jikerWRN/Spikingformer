-----------------------------------------------------------------------
20260720

CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file D_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_del_max -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file D_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file D_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_kernel3 -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file D_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_kernel3 -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1



--------------------------------------------

CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file D_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_kernel3_back_UPB_15_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1


CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file D_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_add_internal_back_UPB_15_online -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1


--------------------------------------------




CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file F_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_kernel3_back_one -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1


CUDA_VISIBLE_DEVICES=0 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file F_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_kernel3_back_two -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1





CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file F_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_kernel3_back_three -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1



CUDA_VISIBLE_DEVICES=1 python cifar10/train_adaptive_count_to_adaptive.py -c cifar10/cifar10.yml --model-file F_model_4layers_baseline_ILIF01_count_to_adaptive_back_count_kernel3_back_four -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/B_output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100 --time-step 1