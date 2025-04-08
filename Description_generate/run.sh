set -ex 

node_index=0
node_num=8
chunk_num=8

# bash prepare.sh

for (( chunk_index=0; chunk_index<=$[$chunk_num-1]; chunk_index++ ))
do
    # Assign each process to a unique GPU using chunk_index
    gpu_id=$chunk_index  # Assuming chunk_index corresponds to a GPU (e.g., 0, 1, 2 for 3 GPUs)
    
    # Set CUDA_VISIBLE_DEVICES to assign a unique GPU to each process
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python3 multi_process.py \
        --input_file divide_result_file \
        --output_folder generate_result_folder  \
        --chunk_index $chunk_index \
        --chunk_num $chunk_num \
        --node_index $node_index \
        --node_num $node_num > /home/pengruotian/patch_matter/description/test_$chunk_index.log 2>&1 &
done

wait

python3 /home/pengruotian/patch_matter/description/combine.py \
    --folder_path above_output_folder \
    --output_file description_output.json
