moses_scripts=/home/meher/nematus/mosesdecoder/scripts
bpe_scripts=/home/meher/nematus/subword-nmt/subword_nmt
nematus_home=/home/meher/nematus/nematus
bpe_operations=5000
bpe_threshold=20

main_dir=/home/meher/model_HiEn/wmt17-transformer-scripts/training
script_dir=$main_dir/scripts
data_dir=$main_dir/data
model_dir=$main_dir/model
src=hi
trg=en

# echo "preprocesing started"
#  # tokenize
#  for prefix in corpus dev test
#   do
#     cat $data_dir/$prefix.$src | \
#     $moses_scripts/tokenizer/normalize-punctuation.perl -l $src | \
#     $moses_scripts/tokenizer/tokenizer.perl -a -l $src > $data_dir/$prefix.tok.$src

#     cat $data_dir/$prefix.$trg | \
#     $moses_scripts/tokenizer/normalize-punctuation.perl -l $trg | \
#     $moses_scripts/tokenizer/tokenizer.perl -a -l $trg > $data_dir/$prefix.tok.$trg

#   done

# # # clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
#  $moses_scripts/training/clean-corpus-n.perl $data_dir/corpus.tok $src $trg $data_dir/corpus.tok.clean 1 80

# # # train truecaser
#  $moses_scripts/recaser/train-truecaser.perl -corpus $data_dir/corpus.tok.clean.$src -model $model_dir/truecase-model.$src
#  $moses_scripts/recaser/train-truecaser.perl -corpus $data_dir/corpus.tok.clean.$trg -model $model_dir/truecase-model.$trg

#  # apply truecaser (cleaned training corpus)
#  for prefix in corpus
#   do
#    $moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$src < $data_dir/$prefix.tok.clean.$src > $data_dir/$prefix.tc.$src
#    $moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$trg < $data_dir/$prefix.tok.clean.$trg > $data_dir/$prefix.tc.$trg
#   done

#  $bpe_scripts/learn_joint_bpe_and_vocab.py -i $data_dir/corpus.tok.clean.$src $data_dir/corpus.tok.clean.$trg --write-vocabulary $data_dir/vocab.$src $data_dir/vocab.$trg -s $bpe_operations -o $data_dir/$src$trg.bpe

#  for prefix in corpus.tok.clean dev.tok test
#   do
#  $bpe_scripts/apply_bpe.py -c $data_dir/$src$trg.bpe --vocabulary $data_dir/vocab.$src --vocabulary-threshold $bpe_threshold < $data_dir/$prefix.$src > $data_dir/$prefix.bpe.$src
#  $bpe_scripts/apply_bpe.py -c $data_dir/$src$trg.bpe --vocabulary $data_dir/vocab.$trg --vocabulary-threshold $bpe_threshold < $data_dir/$prefix.$trg > $data_dir/$prefix.bpe.$trg
#   done

#  echo "preprocesing done"

#  echo "build_dictionary now"
#  cat $data_dir/corpus.tok.clean.bpe.$src $data_dir/corpus.tok.clean.bpe.$trg > $data_dir/corpus.tok.clean.bpe.$src.$trg
#  $nematus_home/data/build_dictionary.py $data_dir/corpus.tok.clean.bpe.$src.$trg

#  devices=0,1,2,3
#  echo "started training"

#  CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/train.py \
#      --source_dataset $data_dir/corpus.tok.clean.bpe.$src \
#      --target_dataset $data_dir/corpus.tok.clean.bpe.$trg \
#      --dictionaries $data_dir/corpus.tok.clean.bpe.$src.$trg.json \
#                     $data_dir/corpus.tok.clean.bpe.$src.$trg.json \
#      --save_freq 20000 \
#      --model $model_dir/model \
#      --max_epochs 150\
#      --reload latest_checkpoint \
#      --model_type rnn \
#      --embedding_size 128 \
#      --state_size 128 \
#      --tie_decoder_embeddings \
#      --loss_function per-token-cross-entropy \
#      --label_smoothing 0.1 \
#      --optimizer adam \
#      --adam_beta1 0.9 \
#      --adam_beta2 0.98 \
#      --adam_epsilon 1e-09 \
#      --warmup_steps 4000 \
#      --learning_schedule transformer \
#      --maxlen 100 \
#      --batch_size 16 \
#      --patience 150 \
#      --valid_source_dataset $data_dir/dev.tok.bpe.$src \
#      --valid_target_dataset $data_dir/dev.tok.bpe.$trg \
#      --valid_freq 10000 \
#      --disp_freq 1000 \
#      --sample_freq 0 \
#      --beam_freq 0 \
#      --beam_size 4 \
#      --translation_maxlen 100 \
#      --normalization_alpha 0.6

#  echo "training done"

# model=/home/meher/model_HiEn/wmt17-transformer-scripts/training/model/
model=$model_dir/model-440000
echo "giving outputs"
CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/translate.py \
     -m $model \
     -i $data_dir/test.bpe.$src \
     -o $model_dir/test.output.440000.$trg \
     -k 12 \
     -n 0.6 \
     -b 10

model_new=/home/meher/model_HiEn/wmt17-transformer-scripts/training/model/
# postproces
$script_dir/postprocess.sh < $model_new/test.output.440000.$trg > $model_new/test.output.postprocessed.440000.$trg

# postprocess (no detokenization)
$script_dir/postprocess_tokenized.sh < $model_new/test.output.440000.$trg > $model_new/test.output.tokenized.440000.$trg

# evaluate with detokenized BLEU (same as mteval-v13a.pl)
echo "$test_prefix (detokenized BLEU)"
$nematus_home/data/multi-bleu-detok.perl $data_dir/test.$trg < $model_new/test.output.postprocessed.440000.$trg


# Need to tokenise the text too
# tokenize
# echo "Tokebising the test file"
# for prefix in test
#  do
#    cat $data_dir/$prefix.$src | \
#    $moses_scripts/tokenizer/normalize-punctuation.perl -l $src | \
#    $moses_scripts/tokenizer/tokenizer.perl -a -l $src > $data_dir/$prefix.tok.$src

#    cat $data_dir/$prefix.$trg | \
#    $moses_scripts/tokenizer/normalize-punctuation.perl -l $trg | \
#    $moses_scripts/tokenizer/tokenizer.perl -a -l $trg > $data_dir/$prefix.tok.$trg

#  done
# evaluate with tokenized BLEU

echo "$test_prefix (tokenized BLEU)"
$nematus_home/data/multi-bleu.perl $data_dir/test.tok.$trg < $model_new/test.output.tokenized.440000.$trg

