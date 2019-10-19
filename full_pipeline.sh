#./full_pipline.sh eng_bho 200d/e2b eng bho 200 200
# moses_scripts=/home/saumitra/github_packages/mosesdecoder/scripts
bpe_scripts=/home/saumitra/github_packages/subword-nmt
nematus_home=/home/saumitra/github_packages/nematus/
bpe_operations=5000
bpe_threshold=20

main_dir=/home/saumitra/loresmt
data_dir=$main_dir/$1
model_dir=$main_dir/$1/$2
src=$3
trg=$4
echo "preprocesing started"

$bpe_scripts/learn_joint_bpe_and_vocab.py -i $data_dir/train.tok.clean.$src $data_dir/train.tok.clean.$trg --write-vocabulary $data_dir/vocab.$src $data_dir/vocab.$trg -s $bpe_operations -o $data_dir/$src$trg.bpe

for prefix in train.tok.clean dev.tok test te
 do
$bpe_scripts/apply_bpe.py -c $data_dir/$src$trg.bpe --vocabulary $data_dir/vocab.$src --vocabulary-threshold $bpe_threshold < $data_dir/$prefix.$src > $data_dir/$prefix.bpe.$src
$bpe_scripts/apply_bpe.py -c $data_dir/$src$trg.bpe --vocabulary $data_dir/vocab.$trg --vocabulary-threshold $bpe_threshold < $data_dir/$prefix.$trg > $data_dir/$prefix.bpe.$trg
 done

echo "preprocesing done"

echo "build_dictionary now"
cat $data_dir/train.tok.clean.bpe.$src $data_dir/train.tok.clean.bpe.$trg > $data_dir/train.tok.clean.bpe.$src.$trg
$nematus_home/data/build_dictionary.py $data_dir/train.tok.clean.bpe.$src.$trg

devices=0,1,2,3
echo "started training"

CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/train.py \
    --source_dataset $data_dir/train.tok.clean.bpe.$src \
    --target_dataset $data_dir/train.tok.clean.bpe.$trg \
    --dictionaries $data_dir/train.tok.clean.bpe.$src.$trg.json \
                   $data_dir/train.tok.clean.bpe.$src.$trg.json \
    --save_freq 20000 \
    --model $model_dir/model \
    --reload latest_checkpoint \
    --model_type rnn \
    --embedding_size $5 \
    --state_size $6 \
    --tie_decoder_embeddings \
    --loss_function per-token-cross-entropy \
    --label_smoothing 0.1 \
    --optimizer adam \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-09 \
    --warmup_steps 4000 \
    --learning_schedule transformer \
    --maxlen 100 \
    --batch_size 200 \
    --valid_source_dataset $data_dir/dev.tok.bpe.$src \
    --valid_target_dataset $data_dir/dev.tok.bpe.$trg \
	--valid_batch_size 100 \
    --valid_freq 10000 \
    --disp_freq 1000 \
    --sample_freq 0 \
    --beam_freq 0 \
    --beam_size 4 \
    --translation_maxlen 100 \
    --normalization_alpha 0.6

echo "training done"

model=$model_dir/model
echo "giving outputs"
CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/translate.py \
     -m $model \
     -i $data_dir/test.bpe.$src \
     -o $model_dir/test.output.$trg \
     -k 12 \
     -n 0.6 \
     -b 10

CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/translate.py \
     -m $model \
     -i $data_dir/te.bpe.$src \
     -o $model_dir/te.output.$trg \
     -k 12 \
     -n 0.6 \
     -b 10

