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
        --input_file /test_data/did_bench.json \
        --output_folder /description_generate \
        --chunk_index $chunk_index \
        --chunk_num $chunk_num \
        --node_index $node_index \
        --node_num $node_num > description/test_$chunk_index.log 2>&1 &
done

wait

python3 /description_generate/combine.py \
    --folder_path /description_generate \
    --output_file /description_generate/description_output.json
